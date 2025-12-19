from __future__ import annotations

import importlib
import io
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import joblib
from typing_extensions import Self

from modularml.core.data.schema_constants import MML_EXTENSION, MML_FILE_VERSION
from modularml.utils.backend import Backend
from modularml.utils.optional_imports import ensure_torch

# Map each class to a unique file extension
# class -> short kind string (e.g. FeatureSet -> "fs")
_CLASS_TO_KIND: dict[type, str] = {}
_KIND_TO_CLASS: dict[str, type] = {}


def register_serializable(
    cls: type,
    *,
    kind: str,
) -> None:
    """
    Register a class for ModularML serialization.

    Enforces:
      - one-to-one mapping between class <-> kind
      - kind uniqueness
    """
    if cls in _CLASS_TO_KIND:
        msg = f"Class {cls.__name__} already registered with kind '{_CLASS_TO_KIND[cls]}'"
        raise ValueError(msg)

    if kind in _KIND_TO_CLASS:
        msg = f"Serialization kind '{kind}' already registered to class {_KIND_TO_CLASS[kind].__name__}."
        raise ValueError(msg)

    if "." in kind:
        msg = f"Class-based extension cannot include '.'. Received: '{kind}'"
        raise ValueError(msg)

    _CLASS_TO_KIND[cls] = kind
    _KIND_TO_CLASS[kind] = cls


def _get_suffix_for_class(cls_name: str) -> str:
    return f".{_CLASS_TO_KIND[cls_name]}{MML_EXTENSION}"


class SerializableMixin(ABC):
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

    # ================================================
    # Filename handling
    # ================================================
    def _normalize_save_path(self, path: pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)

        # Normalize "self" to class name (e.g., "FeatureSet"), not instance
        cls_name: str = self.__name__ if isinstance(self, type) else self.__class__.__qualname__
        exp_suffix: str = _get_suffix_for_class(cls_name)
        if not path.name.endswith(exp_suffix):
            return path.with_suffix("").with_suffix(exp_suffix)
        return path

    # ================================================
    # Structured State
    # ================================================
    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Returns the complete internal state as a pure-Python dictionary.

        Must be fully reconstructable via set_state().
        Must contain version information.
        """
        ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the internal state from a dictionary produced by get_state()."""
        ...

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

        This include the state and large objects \
        (e.g., Arrow tables, weight dicts)
        """
        payload = {
            "__mml__": {
                "version": MML_FILE_VERSION,
                "kind": _CLASS_TO_KIND[self.__class__.__qualname__],
                "class": self.__class__.__qualname__,
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
        if meta["kind"] != _CLASS_TO_KIND[cls.__name__] or meta["class"] != cls.__name__:
            msg = f"File contains data for a {meta['class']} but loader expects {cls.__name__}."
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


class TorchSerializableMixin(SerializableMixin):
    """
    A lightweight reusable serialization mixin for torch models.

    Description:
        The model must already have:
            - `self.config`: a dict of constructor kwargs
            - `self._input_shape`
            - `self._output_shape`

        This mixin only handles the generic logic. Your model must:
            - Provide a constructor `__init__(..., **config)`
            - Provide `.config` dict describing the model
            - Implement `build(input_shape, output_shape)`
    """

    # Get structured state
    def get_state(self) -> dict:
        return {
            "class": self.__class__.__module__ + "." + self.__class__.__qualname__,
            "config": self.config,  # constructor args
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "weights": self.get_weights(),  # numpy arrays
        }

    def set_state(self, state: dict):
        # Step 1: reconstruct model
        self.config = state["config"]
        self._input_shape = state["input_shape"]
        self._output_shape = state["output_shape"]

        # Build internal torch layers
        self.build(self._input_shape, self._output_shape)

        # Step 2: load weights
        self.set_weights(state["weights"])

    @classmethod
    def from_state(cls, state: dict):
        from modularml.models.base_model import BaseModel

        torch = ensure_torch()

        # Dynamically import class
        full_name = state["class"]
        module_name, class_name = full_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        klass = getattr(module, class_name)

        # Create empty instance
        obj = klass.__new__(klass)

        # Initialize base classes manually
        torch.nn.Module.__init__(obj)
        BaseModel.__init__(obj, backend=Backend.TORCH)

        # Restore state
        obj.set_state(state)
        return obj

    # Weight handling
    def get_weights(self) -> dict:
        if not self.is_built:
            return {}
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        torch = ensure_torch()
        torch_state = {k: torch.tensor(v) for k, v in weights.items()}
        self.load_state_dict(torch_state, strict=True)
