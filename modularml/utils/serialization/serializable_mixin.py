from __future__ import annotations

import io
import pathlib
from typing import Any

import joblib
from typing_extensions import Self

from modularml.core.data.schema_constants import MML_EXTENSION, MML_FILE_VERSION, MML_STATE_TARGET

# Map each class to a unique file extension
# class -> short kind string (e.g. FeatureSet -> "fs")
_BASE_CLASS_TO_KIND: dict[type, str] = {}
_KIND_TO_BASE_CLASS: dict[str, type] = {}


def register_serializable(
    base_cls: type,
    *,
    kind: str,
) -> None:
    """
    Register a class for ModularML serialization.

    Enforces:
      - one-to-one mapping between class <-> kind
      - kind uniqueness
    """
    if base_cls in _BASE_CLASS_TO_KIND:
        msg = f"Class {base_cls.__name__} already registered with kind '{_BASE_CLASS_TO_KIND[base_cls]}'"
        raise ValueError(msg)

    if kind in _KIND_TO_BASE_CLASS:
        msg = f"Serialization kind '{kind}' already registered to class {_KIND_TO_BASE_CLASS[kind].__name__}."
        raise ValueError(msg)

    if "." in kind:
        msg = f"Class-based extension cannot include '.'. Received: '{kind}'"
        raise ValueError(msg)

    _BASE_CLASS_TO_KIND[base_cls] = kind
    _KIND_TO_BASE_CLASS[kind] = base_cls


def _get_kind_for_class(cls: type) -> str:
    for base, kind in _BASE_CLASS_TO_KIND.items():
        if issubclass(cls, base):
            return kind
    msg = f"No serialization kind registered for class hierarchy of {cls.__name__}"
    raise KeyError(msg)


def _get_suffix_for_class(cls: type) -> str:
    return f".{_get_kind_for_class(cls=cls)}{MML_EXTENSION}"


class SerializableMixin:
    """
    Standardized 3-level serialization interface for all ModularML core classes.

    Level 1 (State):
        - get_state()  -> dict
        - set_state(state_dict)

    Level 2 (Bytes):
        - to_bytes()   -> bytes
        - from_bytes(blob) -> instance

    Level 3 (File I/O):
        - save(path)
        - load(path)

    Classes using this mixin must implement:
        - __init__(...) with enough args to construct an empty shell
        - get_state()
        - set_state(state_dict)
    """

    def __eq__(self, value):
        if not isinstance(value, SerializableMixin):
            raise NotImplementedError
        return self.get_state() == value.get_state()

    def __hash__(self):
        return hash(self.get_state())

    # ================================================
    # Filename handling
    # ================================================
    def _normalize_save_path(self, path: pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)

        # Normalize "self" to class, not instance
        cls: type = self if isinstance(self, type) else self.__class__
        exp_suffix: str = _get_suffix_for_class(cls)
        if not path.name.endswith(exp_suffix):
            return path.with_suffix("").with_suffix(exp_suffix)
        return path

    # ================================================
    # Structured State
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Returns the complete internal state as a pure-Python dictionary.

        Must be fully reconstructable via set_state().
        Must contain version information.
        """
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the internal state from a dictionary produced by get_state()."""
        # Call next set_state in MRO *only if it exists*
        parent = super()
        if hasattr(parent, "set_state"):
            parent.set_state(state)

    @classmethod
    def from_state(cls, state: dict):
        from modularml.core.experiment.experiment_context import ExperimentContext
        from modularml.core.experiment.experiment_node import ExperimentNode

        obj = cls.__new__(cls)  # bypass __init__
        obj.set_state(state)

        if isinstance(obj, ExperimentNode):
            ExperimentContext.register_experiment_node(obj)
        return obj

    # ================================================
    # Bytes
    # ================================================
    def to_bytes(self) -> bytes:
        """
        Serialize state into a single binary artifact.

        This include the state and large objects
        (e.g., Arrow tables, weight dicts)
        """
        payload = {
            "__mml__": {
                "version": MML_FILE_VERSION,
                "kind": _get_kind_for_class(self.__class__),
                MML_STATE_TARGET: f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            },
            "state": self.get_state(),
        }
        buffer = io.BytesIO()
        joblib.dump(payload, buffer, compress=("zlib", 3))
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> Self:
        """
        Reconstruct an object from a blob created by to_bytes().

        This must be overridden in classes with __init__ methods \
        that require arguments.
        """
        # Load dict from bytes
        buffer = io.BytesIO(blob)
        payload = joblib.load(buffer)

        # Validate file data
        if "__mml__" not in payload:
            raise ValueError("Invalid ModularML file received.")
        meta = payload["__mml__"]
        if meta["kind"] != _get_kind_for_class(cls):
            msg = f"File contains kind '{meta['kind']}' but loader expects kind '{_get_kind_for_class(cls)}'."
            raise TypeError(msg)

        # Use from_state to create
        return cls.from_state(payload["state"])

    # ================================================
    # File I/O
    # ================================================
    def save(self, path: str | pathlib.Path, *, overwrite: bool = False) -> None:
        """Save the object to a file using the Level 2 binary format."""
        path = self._normalize_save_path(path)
        if path.exists() and not overwrite:
            msg = f"File already exists: {path}"
            raise FileExistsError(msg)

        blob = self.to_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(blob)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Self:
        """Load an object from a file created via save()."""
        path = pathlib.Path(path)
        if not path.exists():
            msg = f"No such file: {path}"
            raise FileNotFoundError(msg)

        with path.open("rb") as f:
            blob = f.read()

        return cls.from_bytes(blob)
