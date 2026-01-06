from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from .scaler import Scaler
from modularml.scalers import scaler_naming_fn, scaler_registry


def register_builtin():
    symbol_registry.register_builtin_class(
        key="Scaler",
        cls=Scaler,
    )

    symbol_registry.register_builtin_registry(
        import_path="modularml.scalers.scaler_registry",
        registry=scaler_registry,
        naming_fn=scaler_naming_fn,
    )


def register_kinds():
    kind_registry.register(
        cls=Scaler,
        kind=SerializationKind(name="Scaler", kind="sc"),
    )
