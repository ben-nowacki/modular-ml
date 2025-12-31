from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from modularml.core.io.protocols import Configurable, Stateful

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext

T = TypeVar("T")


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
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (T):
                Object instance to encode.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

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
        ctx: SaveContext,
        config_rel_path: str = "config.json",
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (T):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not self.has_config(obj):
            return {"config": None}

        if not hasattr(obj, "get_config"):
            raise NotImplementedError("Object must implement a `get_config` method.")

        config = obj.get_config()

        # Save config to file
        path = self._write_json(config, Path(save_dir) / config_rel_path)
        return {"config": path.name}

    def encode_state(
        self,
        obj: T,
        save_dir: Path,
        *,
        ctx: SaveContext,
        state_rel_path: str = "state.pkl",
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (T):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

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

    # ================================================
    # Object decoding
    # ================================================
    def decode(
        self,
        cls: type[T],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> T:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[T]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            T: The re-instantiated object.

        """
        config = self.decode_config(load_dir=load_dir, ctx=ctx)
        obj = cls(**config)
        if hasattr(obj, "set_state"):
            state = self.decode_state(load_dir=load_dir, ctx=ctx)
            obj.set_state(**state)
        return obj

    def decode_config(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
        config_rel_path: str = "config.json",
    ) -> dict[str, Any] | None:
        """
        Decodes config from a json file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.
            config_rel_path (str):
                Relative path to saved config file.
                Defaults to "config.json"

        Returns:
            dict[str, Any] | None: The decoded config data.

        """
        # Check that config.json exists
        file_config = Path(load_dir) / config_rel_path
        if not file_config.exists():
            msg = f"Could not find config file in directory: '{file_config}'."
            raise FileNotFoundError(msg)

        # Read config
        config = self._read_json(file_config)
        return config

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
        state_rel_path: str = "state.pkl",
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.
            state_rel_path (str):
                Relative path to save state to.
                Defaults to "state.pkl"

        Returns:
            dict[str, Any]: The decoded state data.

        """
        import pickle

        # Check that config.json exists
        file_state = Path(load_dir) / state_rel_path
        if not file_state.exists():
            msg = f"Could not find state file in directory: '{file_state}'."
            raise FileNotFoundError(msg)

        # Read config
        with Path.open(file_state, "rb") as f:
            state: dict[str, Any] = pickle.load(f)

        return state

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
