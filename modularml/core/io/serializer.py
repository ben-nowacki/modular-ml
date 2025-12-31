from __future__ import annotations

import inspect
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from modularml.core.io.artifacts import Artifact, ArtifactHeader
from modularml.core.io.class_registry import ClassRegistry, ClassResolutionError, class_registry
from modularml.core.io.class_spec import ClassSpec
from modularml.core.io.conventions import KindRegistry, kind_registry
from modularml.core.io.handlers.registry import handler_registry
from modularml.core.io.migrations.registry import migration_registry
from modularml.core.io.packaged_code_loaders.default_loader import default_packaged_code_loader
from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.io.handlers.handler import HandlerRegistry


def _zip_dir(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))


def _unzip(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)


def _enforce_file_suffix(path: Path, cls: type) -> Path:
    # Enforce 'kind.mml' suffix
    path = Path(path)
    exp_suffix = kind_registry.get_kind(cls=cls).file_suffix
    if "".join(path.suffixes) != exp_suffix:
        path = path.with_name(path.stem.split(".")[0] + exp_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


class SaveContext:
    """
    Context provided to handlers during save so they can request ClassSpecs and bundling.

    Args:
        artifact_path (Path): Root folder of the artifact being written.
        policy (SerializationPolicy): Root policy for the object being saved.
        serializer (Any): Serializer instance with make_class_spec helpers.

    """

    def __init__(
        self,
        artifact_path: Path,
        serializer: Serializer,
        mml_version: str = "1.0.0",
    ):
        self.artifact_path = artifact_path
        self.serializer = serializer
        self.mml_version = mml_version

        self._packaged_classes: dict[type, ClassSpec] = {}

    def package_class(self, cls: type) -> ClassSpec:
        """Ensure `cls` is packaged into the artifact and return its ClassSpec."""
        if cls in self._packaged_classes:
            return self._packaged_classes[cls]

        # We need to copy the source code (entire file) into an artifact
        code_dir = self.artifact_path / "code"
        code_dir.mkdir(exist_ok=True)

        # If was previously packaged, we don't need to inspect (the source text is stored in the object)
        dest_file = None
        if hasattr(cls, "__mml_source_text__") and hasattr(cls, "__mml_source_ref__"):
            src_file = str(cls.__mml_source_ref__).split(":")[0]
            dest_file = code_dir / Path(src_file).name
            if dest_file.exists():
                msg = f"File for packaged code already exists: {dest_file}"
                raise FileExistsError(msg)

            # Can copy the source text directly
            dest_file.write_text(str(cls.__mml_source_text__))

        # Otherwise, we need to inspect the file
        else:
            # Block __main__ classes
            if cls.__module__ == "__main__":
                msg = (
                    f"Cannot package class '{cls.__qualname__}' defined in '__main__'. "
                    f"Packaged serialization requires the class to be defined in a standalone Python file."
                )
                raise RuntimeError(msg)

            # Resolve source file
            try:
                src_file = Path(inspect.getfile(cls))
            except TypeError as exc:
                msg = f"Cannot bundle source for class '{cls.__name__}'. The class does not have a resolvable source file."
                raise RuntimeError(msg) from exc

            # Check that source file exists
            if not src_file.exists():
                msg = f"Source file not found: {src_file}"
                raise FileNotFoundError(msg)

            # Copy file contents
            dest_file = code_dir / src_file.name
            if dest_file.exists():
                msg = f"File for packaged code already exists: {dest_file}"
                raise FileExistsError(msg)
            shutil.copy2(src_file, dest_file)

        # Build source_ref
        source_ref = f"code/{dest_file.name}:{cls.__qualname__}"

        class_spec = class_registry.identify_class(
            cls=cls,
            policy=SerializationPolicy.PACKAGED,
            source_ref=source_ref,
            key=None,
        )
        self._packaged_classes[cls] = class_spec
        return class_spec

    def make_class_spec(self, cls: type, policy: SerializationPolicy, builtin_key: str) -> ClassSpec:
        """
        Construct or retrieves a ClassSpec for a given class.

        Args:
            cls (type):
                Class to identify.
            policy (SerializationPolicy):
                Class resolution policy.
            builtin_key (str | None):
                Required when policy == BUILTIN.

        Returns:
            ClassSpec of provided class.

        """
        from modularml.core.io.class_registry import class_registry

        policy = normalize_policy(policy)

        if policy is SerializationPolicy.PACKAGED:
            spec = self.package_class(cls=cls)

        else:
            spec = class_registry.identify_class(
                cls,
                policy=policy,
                key=builtin_key,
                source_ref=None,
            )
        spec.validate()
        return spec

    def write_artifact(self, obj: Any, cls_spec: ClassSpec, save_dir: Path):
        """
        Write subartifact to save_dir.

        Args:
            obj (Any):
                Object to serialize.
            cls_spec (ClassSpec):
                ClassSpec of the given object. Use the `make_class_spec` method.
            save_dir (Path):
                Directory to save encodings into.

        Return:
            Path: file path of saved artifact.

        """
        # Get handler for this object type
        handler = handler_registry.resolve(type(obj))

        # Encode config and state files with handler
        file_mapping = handler.encode(obj=obj, save_dir=save_dir, ctx=self)

        # Create artifact for root
        artifact = Artifact(
            header=ArtifactHeader(
                mml_version=self.mml_version,
                schema_version=Artifact.schema_version,
                object_version=handler.object_version,
                kind=kind_registry.get_kind(type(obj)).kind,
                class_spec=asdict(cls_spec),
            ),
            files=file_mapping,
        )

        # Write artifact to disk (as json)
        with (save_dir / "artifact.json").open("w", encoding="utf-8") as f:
            json.dump(artifact.to_json(), f, indent=2, sort_keys=True)

    def emit_mml(self, obj: Any, cls_spec: ClassSpec, out_path: Path, *, overwrite: bool = False) -> Path:
        """
        Write subartifact to out_path.

        Args:
            obj (Any):
                Object to serialize.
            cls_spec (ClassSpec):
                ClassSpec of the given object. Use the `make_class_spec` method.
            out_path (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool):
                If True, overwrites existing artifact at out_path.

        Return:
            Path: file path of saved artifact.

        """
        # Enforce 'kind.mml' suffix
        out_path = _enforce_file_suffix(out_path, cls=obj.__class__)
        if out_path.exists() and not overwrite:
            msg = f"Artifact already exists at: {out_path}"
            raise FileExistsError(msg)

        # Write encodings and artifact in temp directory, then compress to zip
        with TemporaryDirectory() as subtmp:
            subdir = Path(subtmp)

            # Write artifact for this obj
            # This internally calls handler.encode on subdir
            self.write_artifact(obj=obj, cls_spec=cls_spec, save_dir=subdir)

            # Compress subdir to a zip file and save to out_path
            _zip_dir(subdir, out_path)

        return out_path


class LoadContext:
    """
    Context provided to TypeHandlers during deserialization.

    Attributes:
        artifact_path (Path):
            Root directory of the artifact being loaded.
        allow_packaged_code (bool):
            Whether executing bundled code is permitted.
        packaged_code_loader (Callable):
            Loader used to execute bundled code for PACKAGED classes.
        serializer (Any):
            Owning serializer instance (for advanced resolution if needed).

    """

    def __init__(
        self,
        artifact_path: Path,
        serializer: Serializer,
        *,
        allow_packaged_code: bool,
    ):
        self.artifact_path = artifact_path
        self.serializer = serializer
        self.allow_packaged_code = allow_packaged_code

    def load_from_dir(
        self,
        dir_load: Path,
        packaged_code_loader: Callable[[str], object] | None = None,
        provided_cls: type | None = None,
    ):
        """Load an artifact from an unzipped directory."""
        if packaged_code_loader is None:

            def packaged_code_loader(source_ref):
                return default_packaged_code_loader(
                    artifact_path=dir_load,
                    source_ref=source_ref,
                    allow_packaged=self.allow_packaged_code,
                )

        # Validate artifact existence
        artifact_json = dir_load / "artifact.json"
        if not artifact_json.exists():
            raise FileNotFoundError("artifact.json missing from archive")

        # Read artifact
        with artifact_json.open("r", encoding="utf-8") as f:
            artifact = Artifact.from_json(json.load(f))

        # Extract ClassSpec
        spec = ClassSpec(**artifact.header.class_spec)
        spec.validate()

        # Resolve class (no object instantiation yet)
        if spec.policy is SerializationPolicy.STATE_ONLY:
            if provided_cls is None:
                raise ClassResolutionError("STATE_ONLY artifacts require `provided_cls`.")
            cls = provided_cls
        else:
            cls = class_registry.resolve_class(
                spec,
                allow_packaged_code=self.allow_packaged_code,
                packaged_code_loader=packaged_code_loader,
            )

        # Apply any registered migrations
        artifact = migration_registry.apply(
            artifact=artifact,
            artifact_path=dir_load,
            handler=handler_registry.resolve(cls=cls),
        )

        # Reconstruct object (handler class handles this logic)
        handler = handler_registry.resolve(cls)
        obj = handler.decode(cls=cls, load_dir=dir_load, ctx=self)

        return obj

    def load_from_mml(
        self,
        mml_file: Path,
        packaged_code_loader: Callable[[str], object] | None = None,
        provided_cls: type | None = None,
    ):
        """Load an artifact from a zipped ('.mml') directory."""
        path: Path = Path(mml_file)
        if not path.exists():
            raise FileNotFoundError(path)

        if packaged_code_loader is None:

            def packaged_code_loader(source_ref):
                return default_packaged_code_loader(
                    artifact_path=mml_file,
                    source_ref=source_ref,
                    allow_packaged=self.allow_packaged_code,
                )

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _unzip(path, tmp_path)

            return self.load_from_dir(
                dir_load=tmp_path,
                packaged_code_loader=packaged_code_loader,
                provided_cls=provided_cls,
            )


class Serializer:
    """
    Central serializer that saves and loads ModularML objects using registries and handlers.

    The serializer:
    - determines artifact kind (KindRegistry)
    - determines class identity (ClassRegistry + SerializationPolicy)
    - encodes config/state (TypeHandlers, Configurable, Stateful)
    - writes to disk and reconstructs on load

    Args:
        kind_registry (KindRegistry):
            Registry for file suffix/kind mapping.
        class_registry (ClassRegistry):
            Registry for ClassSpec to/from class resolution.
        handler_registry (HandlerRegistry):
            Registry for TypeHandlers.
        mml_version (str):
            ModularML version string to embed in artifacts.
        schema_version (int):
            Artifact schema version for migrations.

    """

    def __init__(
        self,
        *,
        kind_registry: KindRegistry,
        class_registry: ClassRegistry,
        handler_registry: HandlerRegistry,
        mml_version: str = "0.0.0",
    ):
        self.kind_registry = kind_registry
        self.class_registry = class_registry
        self.handler_registry = handler_registry
        self.mml_version = mml_version

    # ================================================
    # Save
    # ================================================
    def save(
        self,
        obj: Any,
        save_path: str,
        *,
        policy: SerializationPolicy,
        builtin_key: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Save an object to disk as a ModularML artifact.

        Args:
            obj (Any):
                Object to serialize.
            save_path (str):
                Output directory save_path for the artifact.
            policy (SerializationPolicy):
                Class resolution policy.
            builtin_key (str | None):
                Required when policy == BUILTIN.
            overwrite (bool):
                If True, overwrites existing artifact directory.

        Returns:
            str: Path to the artifact directory.

        """
        # Enforce 'kind.mml' suffix
        save_path = _enforce_file_suffix(save_path, cls=obj.__class__)
        if save_path.exists() and not overwrite:
            msg = f"Artifact already exists at: {save_path}"
            raise FileExistsError(msg)

        # Wrap everything in temp directory
        # This gets cast into a zip file at the end
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create SaveContext for any recursive calls
            ctx = SaveContext(
                artifact_path=tmp_path,
                serializer=self,
                mml_version=self.mml_version,
            )

            # Write artifact
            cls_spec = ctx.make_class_spec(
                cls=obj.__class__,
                policy=policy,
                builtin_key=builtin_key,
            )
            ctx.write_artifact(
                obj=obj,
                cls_spec=cls_spec,
                save_dir=tmp_path,
            )

            # Compress tmp_dir to a zip file and save to save_path
            _zip_dir(tmp_path, save_path)

        return str(save_path)

    # ================================================
    # Load
    # ================================================
    def load(
        self,
        path: str,
        *,
        allow_packaged_code: bool = False,
        provided_class: type | None = None,
    ) -> Any:
        """
        Load an artifact from disk and reconstruct the serialized object.

        Args:
            path (str): Artifact directory.
            allow_packaged_code (bool): Whether bundled code execution is allowed.
            packaged_code_loader (Any): Callable used for PACKAGED fallback loading.
            provided_class (type | None): Required when policy == STATE_ONLY.

        Returns:
            Any: Reconstructed object.

        """
        import modularml.preprocessing  # noqa: F401

        path: Path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _unzip(path, tmp_path)

            # Create LoadContext for any recursive calls
            ctx = LoadContext(
                artifact_path=tmp_path,
                serializer=self,
                allow_packaged_code=allow_packaged_code,
            )

            return ctx.load_from_dir(
                dir_load=tmp_path,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=tmp_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
                provided_cls=provided_class,
            )


serializer = Serializer(
    kind_registry=kind_registry,
    class_registry=class_registry,
    handler_registry=handler_registry,
    mml_version="1.0.0",
)
