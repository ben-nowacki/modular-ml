from __future__ import annotations

import inspect
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modularml.core.io.artifacts import Artifact, ArtifactHeader
from modularml.core.io.class_registry import ClassRegistry, ClassResolutionError, class_registry
from modularml.core.io.class_spec import ClassSpec
from modularml.core.io.conventions import KindRegistry, kind_registry
from modularml.core.io.handlers.handler import HandlerRegistry, LoadContext, SaveContext
from modularml.core.io.handlers.registry import handler_registry
from modularml.core.io.migrations.registry import migration_registry
from modularml.core.io.packaged_code_loaders.default_loader import default_packaged_code_loader
from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy


def _zip_dir(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))


def _unzip(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)


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
        policy = normalize_policy(policy)
        save_path: Path = Path(save_path)

        # Enforce 'kind.mml' suffix
        exp_suffix = kind_registry.get_kind(cls=obj.__class__).file_suffix
        if "".join(save_path.suffixes) != exp_suffix:
            save_path = save_path.with_name(save_path.stem.split(".")[0] + exp_suffix)

        if save_path.exists() and not overwrite:
            msg = f"Artifact already exists at: {save_path}"
            raise FileExistsError(msg)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Wrap everything in temp directory
        # This gets cast into a zip file at the end
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Resolve kind + handler
            kind = self.kind_registry.get_kind(type(obj)).kind
            handler = self.handler_registry.resolve(type(obj))

            # Get class identity
            # If policy == packaged, we need to copy code
            cls = type(obj)
            final_source_ref = None
            if policy == SerializationPolicy.PACKAGED:
                final_source_ref = self._package_class_source(cls, tmp_path)

            spec = self.class_registry.identify_class(
                cls,
                policy=policy,
                key=builtin_key,
                source_ref=final_source_ref,
            )
            spec.validate()

            # Config & state encoding
            file_mapping = handler.encode(
                obj=obj,
                save_dir=tmp_path,
                ctx=SaveContext(
                    artifact_path=tmp_path,
                    policy=policy,
                    serializer=self,
                ),
            )

            artifact = Artifact(
                header=ArtifactHeader(
                    mml_version=self.mml_version,
                    schema_version=Artifact.schema_version,
                    object_version=handler.object_version,
                    kind=kind,
                    class_spec=asdict(spec),
                ),
                files=file_mapping,
            )

            # Write artifact.json
            with (tmp_path / "artifact.json").open("w", encoding="utf-8") as f:
                json.dump(artifact.to_json(), f, indent=2, sort_keys=True)

            # Zip the entire artifact directory
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

            # Validate artifact existence
            artifact_json = tmp_path / "artifact.json"
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
                if provided_class is None:
                    raise ClassResolutionError("STATE_ONLY artifacts require provided_class.")
                cls = provided_class
            else:
                cls = self.class_registry.resolve_class(
                    spec,
                    allow_packaged_code=allow_packaged_code,
                    packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                        artifact_path=tmp_path,
                        source_ref=source_ref,
                        allow_packaged=allow_packaged_code,
                    ),
                )

            # Apply any registered migrations
            artifact = migration_registry.apply(
                artifact=artifact,
                artifact_path=tmp_path,
                handler=handler_registry.resolve(cls=cls),
            )

            # Reconstruct object (handler class handles this logic)
            handler = self.handler_registry.resolve(cls)
            obj = handler.decode(
                cls=cls,
                parent_dir=tmp_path,
                ctx=LoadContext(
                    artifact_path=tmp_path,
                    allow_packaged_code=allow_packaged_code,
                    packaged_code_loader=default_packaged_code_loader,
                    serializer=self,
                ),
            )

            return obj

    # ================================================
    # SerializationPolicy Helpers
    # ================================================
    def _package_class_source(
        self,
        cls: type,
        artifact_path: Path,
    ) -> str:
        """
        Bundle the source file defining `cls` into the artifact.

        Returns:
            str: source_ref of the form "code/<filename>.py:<qualname>"

        Raises:
            RuntimeError:
                If the class is defined in __main__ or its source cannot be located.
            FileNotFoundError:
                If the resolved source file does not exist.

        """
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

        # Copy source file (entire file) into an artifact
        code_dir = artifact_path / "code"
        code_dir.mkdir(exist_ok=True)

        dest_file = code_dir / src_file.name
        if dest_file.exists():
            msg = f"File for packaged code already exists: {dest_file}"
            raise FileExistsError(msg)
        shutil.copy2(src_file, dest_file)

        # Build source_ref
        qualname = cls.__qualname__
        source_ref = f"code/{dest_file.name}:{qualname}"

        return source_ref

    def make_class_spec(
        self,
        cls: type,
        *,
        policy: SerializationPolicy,
        artifact_path: Path,
        builtin_key: str | None = None,
    ) -> ClassSpec:
        source_ref = None

        if policy is SerializationPolicy.PACKAGED:
            source_ref = self._package_class_source(cls=cls, artifact_path=artifact_path)

        spec = self.class_registry.identify_class(
            cls,
            policy=policy,
            key=builtin_key,
            source_ref=source_ref,
        )
        spec.validate()
        return spec


serializer = Serializer(
    kind_registry=kind_registry,
    class_registry=class_registry,
    handler_registry=handler_registry,
)
