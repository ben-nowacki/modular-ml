from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.utils.registries import CaseInsensitiveRegistry

SCALER_REGISTRY = CaseInsensitiveRegistry()

from .scaler import Scaler


def register_classes():
    symbol_registry.register_builtin(
        key="Scaler",
        cls=Scaler,
    )


def register_kinds():
    kind_registry.register(
        cls=Scaler,
        kind=SerializationKind(name="Scaler", kind="sc"),
    )
