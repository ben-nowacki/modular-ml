from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

MML_FILE_EXTENSION = "mml"


@dataclass(frozen=True)
class SerializationKind:
    """
    Defines the on-disk identity for a serializable object category.

    Attributes:
        name: Human-readable kind name (e.g., "FeatureSet").
        kind: Short suffix identifier (e.g., "fs").

    """

    name: str
    kind: str

    @property
    def file_suffix(self) -> str:
        return f".{self.kind}.{MML_FILE_EXTENSION}"


class KindRegistry:
    """
    Central registry mapping base classes to serialization kinds.

    Resolution is MRO-based: subclasses inherit the first matching base class kind.
    """

    _registry: ClassVar[dict[type, SerializationKind]] = {}
    _rev_registry: ClassVar[dict[SerializationKind, type]] = {}

    def register(
        self,
        cls: type,
        kind: SerializationKind,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a class as the base for a serialization kind.

        Args:
            cls: Base class to associate with this serialization kind.
            kind: SerializationKind definition.
            overwrite: Allow overwriting existing registrations.

        """
        if "." in kind.kind:
            msg = f"Serialization kind cannot contain '.': '{kind.kind}'"
            raise ValueError(msg)

        if not overwrite and cls in self._registry:
            msg = f"Class {cls.__name__} already registered as '{self._registry[cls].kind}'."
            raise ValueError(msg)

        if not overwrite and kind in self._rev_registry:
            msg = f"Serialization kind '{kind}' already registered to {self._rev_registry[kind].__name__}."
            raise ValueError(msg)

        self._registry[cls] = kind
        self._rev_registry[kind] = cls

    def register_kind(
        self,
        *,
        name: str,
        kind: str,
        overwrite: bool = False,
    ) -> Callable[[type], type]:
        """
        Decorator for registering a class as a serialization base.

        Example:
            @kind_registry.register_kind(name="FeatureSet", kind="fs")
            class FeatureSet:
                ...

        """
        serialization_kind = SerializationKind(name=name, kind=kind)

        def decorator(cls: type) -> type:
            self.register(cls, serialization_kind, overwrite=overwrite)
            return cls

        return decorator

    def get_kind(self, cls: type) -> SerializationKind:
        """
        Resolve the serialization kind for a class using MRO lookup.

        Args:
            cls: Class to resolve.

        Returns:
            SerializationKind associated with the nearest registered base class.

        """
        for base in cls.__mro__:
            if base in self._registry:
                return self._registry[base]
        msg = f"No serialization kind registered for class hierarchy of {cls.__name__}"
        raise KeyError(msg)

    def clear(self):
        """Clears all registered items."""
        self._registry.clear()
        self._rev_registry.clear()


kind_registry = KindRegistry()
