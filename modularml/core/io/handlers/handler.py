from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.io.serialization_policy import SerializationPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.io.class_spec import ClassSpec
    from modularml.core.io.serializer import Serializer

T = TypeVar("T")


@dataclass(frozen=True)
class EncodeResult:
    """
    Result of state encoding.

    Args:
        state_spec (Any):
            A serializable StateSpec-like dict describing stored state.

    """

    state_spec: dict[str, Any]


@dataclass(frozen=True)
class SaveContext:
    """
    Context provided to handlers during save so they can request ClassSpecs and bundling.

    Args:
        artifact_path (Path): Root folder of the artifact being written.
        policy (SerializationPolicy): Root policy for the object being saved.
        serializer (Any): Serializer instance with make_class_spec helpers.

    """

    artifact_path: Path
    policy: SerializationPolicy
    serializer: Serializer

    def package_class(self, cls: type) -> ClassSpec:
        """Ensure `cls` is packaged into the artifact and return its ClassSpec."""
        return self.serializer.make_class_spec(
            cls=cls,
            policy=SerializationPolicy.PACKAGED,
            artifact_path=self.artifact_path,
        )


@dataclass(frozen=True)
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

    artifact_path: Path
    allow_packaged_code: bool
    packaged_code_loader: Callable[[str], object]
    serializer: Serializer


class TypeHandler(Generic[T]):
    """Base handler for encoding/decoding config and state for a family of objects."""

    object_version: ClassVar[str] = "1.0"

    # ================================================
    # Object encoding
    # ================================================
    def encode(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext | None = None,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (T):
                Object instance to encode.
            save_dir (Path):
                Parent dir to save config and state files.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str | None]: Mapping of "config" and "state" keys to saved files.

        """
        file_mapping = self.encode_config(obj=obj, save_dir=save_dir, ctx=ctx)
        file_mapping.update(self.encode_state(obj=obj, save_dir=save_dir, ctx=ctx))
        return file_mapping

    def encode_config(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (T):
                Object to encode config for.
            save_dir (Path):
                Parent dir to save 'config.json' file.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        return self._encode_config_json(obj=obj, save_dir=save_dir, ctx=ctx)

    def encode_state(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (T):
                Object to encode state for.
            save_dir (Path):
                Parent dir to save 'state.pkl' file.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        return self._encode_state_pickle(obj=obj, save_dir=save_dir, ctx=ctx)

    # ================================================
    # Object decoding
    # ================================================
    def decode(
        self,
        cls: type[T],
        parent_dir: Path,
        *,
        ctx: LoadContext | None = None,
    ) -> T:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[T]):
                Load config for class.
            parent_dir (Path):
                Directory contains a saved 'config.json' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            T: The re-instantiated Scaler.

        """
        config = self.decode_config(config_dir=parent_dir, ctx=ctx)
        obj = cls(**config)
        if hasattr(obj, "set_state"):
            state = self.decode_state(state_dir=parent_dir, ctx=ctx)
            obj.set_state(**state)
        return obj

    def decode_config(
        self,
        config_dir: Path,
        *,
        ctx: LoadContext | None = None,
    ) -> dict[str, Any]:
        """
        Decodes config from a json file.

        Args:
            config_dir (Path):
                Directory contains a saved 'config.json' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any]: The decoded config data.

        """
        return self._decode_config_json(config_dir=config_dir, ctx=ctx)

    def decode_state(
        self,
        state_dir: str,
        *,
        ctx: LoadContext | None = None,
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            state_dir (Path):
                Directory containing a saved 'state.pkl' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any]: The decoded state data.

        """
        return self._decode_state_pickle(state_dir=state_dir, ctx=ctx)

    # ================================================
    # Convenience Methods
    # ================================================
    def has_config(self, obj: T) -> bool:
        """
        Return True if object has config that should be serialized.

        Args:
            obj (T): Object to inspect.

        Returns:
            bool: True if Configurable.

        """
        return isinstance(obj, Configurable)

    def has_state(self, obj: T) -> bool:
        """
        Return True if object has state that should be serialized.

        Args:
            obj (T): Object to inspect.

        Returns:
            bool: True if Stateful.

        """
        return isinstance(obj, Stateful)

    # ================================================
    # JSON-based config encode/decode
    # ================================================
    def _encode_config_json(
        self,
        obj: Configurable,
        save_dir: Path,
        *,
        config_rel_path: str = "config.json",
        ctx: SaveContext | None = None,  # noqa: ARG002
    ) -> dict[str, str | None]:
        """
        Encodes config to a json file.

        Args:
            obj (Configurable):
                Configurable object to encode config for.
            save_dir (Path):
                Parent dir to save 'config.json' file.
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str | None]: Mapping of config to saved json file

        """
        if not self.has_config(obj):
            return {"config": None}

        if not hasattr(obj, "get_config"):
            raise NotImplementedError("Object must implement a `get_config` method.")

        config = obj.get_config()

        # Save config to file
        path = self._write_json(config, Path(save_dir) / config_rel_path)
        return {"config": path.name}

    def _decode_config_json(
        self,
        config_dir: str,
        *,
        config_rel_path: str = "config.json",
        ctx: LoadContext | None = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """
        Decodes config from a json file.

        Args:
            config_dir (Path):
                Directory contains a saved 'config.json' file
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any] | None: The decoded config data.

        """
        # Check that config.json exists
        file_config = Path(config_dir) / config_rel_path
        if not file_config.exists():
            msg = f"Could not find config file in directory: '{file_config}'."
            raise FileNotFoundError(msg)

        # Read config
        config = self._read_json(file_config)
        return config

    def _write_json(self, data: dict[str, Any], save_path: Path) -> Path:
        """Saves `data` to `path` as json."""
        path = Path(save_path).with_suffix(".json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def _read_json(self, data_path: Path) -> Any:
        """Reads `data` from `data_path` as json."""
        with Path(data_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    # ================================================
    # Pickle-based state encode/decode
    # ================================================
    def _encode_state_pickle(
        self,
        obj: Stateful,
        save_dir: Path,
        *,
        state_rel_path: str = "state.pkl",
        ctx: SaveContext | None = None,  # noqa: ARG002
    ) -> dict[str, str | None]:
        """
        Encodes stateful object's state to a pickle file.

        Args:
            obj (Stateful):
                Object to encode state for.
            save_dir (Path):
                Parent dir to save 'state.pkl' file.
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str | None]: Mapping of state to saved pkl file

        """
        import pickle

        if not self.has_state(obj):
            return {"state": None}

        if not hasattr(obj, "get_state"):
            raise NotImplementedError("Scaler must implement a `get_state` method.")

        state = obj.get_state()

        # Save state to file
        file_state = Path(save_dir) / state_rel_path
        with Path.open(file_state, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {"state": state_rel_path}

    def _decode_state_pickle(
        self,
        state_dir: str,
        *,
        state_rel_path: str = "state.pkl",
        ctx: LoadContext | None = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """
        Decodes state from a pkl file.

        Args:
            state_dir (Path):
                Directory containing a saved 'state.pkl' file
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any] | None: The decoded state data.

        """
        import pickle

        # Check that config.json exists
        file_state = Path(state_dir) / state_rel_path
        if not file_state.exists():
            msg = f"Could not find state file in directory: '{file_state}'."
            raise FileNotFoundError(msg)

        # Read config
        with Path.open(file_state, "rb") as f:
            state: dict[str, Any] = pickle.load(f)

        return state


class HandlerRegistry:
    """Registry for mapping base classes to TypeHandlers with MRO resolution."""

    def __init__(self):
        self._handlers: dict[type, TypeHandler] = {}

    def register(
        self,
        cls: type,
        handler: TypeHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a handler for a class (typically a base class).

        Args:
            cls (type): Class to register the handler for.
            handler (TypeHandler): Handler instance.
            overwrite (bool): Overwrite existing mapping if True.

        """
        if not overwrite and cls in self._handlers:
            msg = f"Handler already registered for {cls.__name__}."
            raise ValueError(msg)
        self._handlers[cls] = handler

    def resolve(self, cls: type) -> TypeHandler:
        """
        Resolve a handler using MRO search.

        Args:
            cls (type): Class to resolve.

        Returns:
            TypeHandler: Matching handler.

        """
        for base in cls.__mro__:
            if base in self._handlers:
                return self._handlers[base]
        # Default handler fallback
        return TypeHandler()
