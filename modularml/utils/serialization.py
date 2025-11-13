# modularml/utils/serialization.py

from __future__ import annotations

import io
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import joblib
from typing_extensions import Self


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

    # ------------------------------------------------------------------
    # Level 1: Structured State
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Level 2: Bytes
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """
        Serialize state into a single binary artifact.

        This include the state and large objects \
        (e.g., Arrow tables, weight dicts)
        """
        state = self.get_state()
        buffer = io.BytesIO()
        joblib.dump(state, buffer, compress=("zstd", 3))
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> Self:
        """
        Reconstruct an object from a blob created by to_bytes().

        This must be overridden in classes with __init__ methods \
        that require arguments.
        """
        # Step 1: Load dict from bytes
        buffer = io.BytesIO(blob)
        state = joblib.load(buffer)

        # Step 2: Construct an uninitialized shell
        # Assumes __init__ can be called with no args or with minimal args.
        # If your class requires args, override this method.
        obj = cls.__new__(cls)  # bypass __init__

        # Step 3: Restore state
        obj.set_state(state)
        return obj

    # ------------------------------------------------------------------
    # Level 3: File I/O
    # ------------------------------------------------------------------
    def save(self, path: str | pathlib.Path, *, overwrite: bool = False) -> None:
        """Save the object to a file using the Level 2 binary format."""
        path = pathlib.Path(path)
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
