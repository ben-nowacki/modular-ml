from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from modularml.core.io.protocols import Configurable, Stateful

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.io.serialization_policy import SerializationPolicy
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
    serializer: Any


class TypeHandler(Generic[T]):
    """Base handler for encoding/decoding config and state for a family of objects."""

    def encode_config(self, obj: T, *, ctx: SaveContext | None = None) -> dict[str, Any]:  # noqa: ARG002
        """
        Encode object configuration required for reconstruction.

        Args:
            obj (T): Object to encode.
            ctx (SaveContext): Optional context (reference to Serializer) to adjust
                config for serialization (required for packaged code).

        Returns:
            dict[str, Any]: JSON-serializable config.

        """
        if isinstance(obj, Configurable):
            return obj.get_config()
        msg = f"{type(obj).__name__} is not Configurable and has no handler override."
        raise TypeError(msg)

    def decode_config(self, cls: type[T], config: dict[str, Any], *, ctx: LoadContext | None = None) -> T:  # noqa: ARG002
        """
        Construct object from config.

        Args:
            cls (type[T]): Target class.
            config (dict[str, Any]): JSON-serializable config.
            ctx (LoadContext): Optional context (reference to Serializer) to adjust
                config for deserialization (required for packaged code).

        Returns:
            T: Constructed object.

        """
        if issubclass(cls, Configurable):  # type: ignore[arg-type]
            return cls.from_config(config)  # type: ignore[return-value]
        msg = f"{cls.__name__} is not Configurable and has no handler override."
        raise TypeError(msg)

    def has_state(self, obj: T) -> bool:
        """
        Return True if object has state that should be serialized.

        Args:
            obj (T): Object to inspect.

        Returns:
            bool: True if Stateful.

        """
        return isinstance(obj, Stateful)

    def encode_state(self, obj: T, state_dir: str) -> dict[str, Any] | None:
        """
        Encode object state to disk and return a serializable state spec.

        Args:
            obj (T): Object to encode.
            state_dir (str): Directory where state blobs should be written.

        Returns:
            dict[str, Any] | None: StateSpec-like dict or None if no state.

        """
        if not isinstance(obj, Stateful):
            return None
        # Default encoding: JSON-only state via a handler override if needed.
        # Default to pickle (works for most)
        return self._encode_state_pickle(obj, state_dir)

    def decode_state(self, obj: T, state_dir: str, state_spec: dict[str, Any]) -> None:
        """
        Restore object state from disk.

        Args:
            obj (T): Object instance to restore.
            state_dir (str): Directory containing state blobs.
            state_spec (dict[str, Any]): StateSpec-like dict describing stored state.

        """
        self._decode_state_pickle(obj, state_dir, state_spec)

    # ================================================
    # Default encoding (pickle) - override per backend/type
    # ================================================
    def _encode_state_pickle(self, obj: Stateful, state_dir: str) -> dict[str, Any]:
        import pickle

        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        with Path.open(path / "state.pkl", "wb") as f:
            pickle.dump(obj.get_state(), f)

        return {"format": "pickle", "files": {"state": "state.pkl"}}

    def _decode_state_pickle(self, obj: Stateful, state_dir: str, state_spec: dict[str, Any]) -> None:
        import pickle

        fmt = state_spec.get("format")
        if fmt != "pickle":
            msg = f"Unsupported default state format: {fmt}"
            raise ValueError(msg)

        path = Path(state_dir) / state_spec["files"]["state"]
        with Path.open(path, "rb") as f:
            state = pickle.load(f)

        obj.set_state(state)


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
