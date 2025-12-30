from modularml.core.io.class_registry import class_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.utils.registries import CaseInsensitiveRegistry

SPLITTER_REGISTRY = CaseInsensitiveRegistry()

from .base_splitter import BaseSplitter
from .condition_splitter import ConditionSplitter
from .random_splitter import RandomSplitter


# Register splitters (after imports)
SPLITTER_REGISTRY.update({"RandomSplitter": RandomSplitter, "ConditionSplitter": ConditionSplitter})


def register_classes():
    class_registry.register_builtin(
        key="BaseSplitter",
        cls=BaseSplitter,
    )
    class_registry.register_builtin(
        key="RandomSplitter",
        cls=RandomSplitter,
    )
    class_registry.register_builtin(
        key="ConditionSplitter",
        cls=ConditionSplitter,
    )


def register_kinds():
    kind_registry.register(
        cls=BaseSplitter,
        kind=SerializationKind(name="BaseSplitter", kind="sp"),
    )
