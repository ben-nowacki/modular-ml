from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry

from .featureset import FeatureSet


def register_classes():
    symbol_registry.register_builtin(
        key="FeatureSet",
        cls=FeatureSet,
    )


def register_kinds():
    kind_registry.register(
        cls=FeatureSet,
        kind=SerializationKind(name="FeatureSet", kind="fs"),
    )
