from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.core.training.loss import Loss

from .optimizer import Optimizer


def register_builtin():
    # Register base classes
    symbol_registry.register_builtin_class(
        key="Optimizer",
        cls=Optimizer,
    )
    symbol_registry.register_builtin_class(
        key="Loss",
        cls=Loss,
    )


def register_kinds():
    kind_registry.register(
        cls=Optimizer,
        kind=SerializationKind(name="Optimizer", kind="op"),
    )
    kind_registry.register(
        cls=Loss,
        kind=SerializationKind(name="Loss", kind="ls"),
    )
