from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.conventions import SerializationKind, kind_registry
from modularml.utils.registries import CaseInsensitiveRegistry

SAMPLER_REGISTRY = CaseInsensitiveRegistry()

from .similiarity_condition import SimilarityCondition
from .base_sampler import BaseSampler
from .simple_sampler import SimpleSampler
from .paired_sampler import PairedSampler
from .triplet_sampler import TripletSampler
from .n_sampler import NSampler


# Register samplers (after imports)
SAMPLER_REGISTRY.update(
    {
        "SimpleSampler": SimpleSampler,
        "PairedSampler": PairedSampler,
        "TripletSampler": TripletSampler,
        "NSampler": NSampler,
    },
)


def register_classes():
    symbol_registry.register_builtin(
        key="SimilarityCondition",
        cls=SimilarityCondition,
    )

    symbol_registry.register_builtin(
        key="BaseSampler",
        cls=BaseSampler,
    )
    symbol_registry.register_builtin(
        key="SimpleSampler",
        cls=SimpleSampler,
    )
    symbol_registry.register_builtin(
        key="PairedSampler",
        cls=PairedSampler,
    )
    symbol_registry.register_builtin(
        key="TripletSampler",
        cls=TripletSampler,
    )
    symbol_registry.register_builtin(
        key="NSampler",
        cls=NSampler,
    )


def register_kinds():
    kind_registry.register(
        cls=SimilarityCondition,
        kind=SerializationKind(name="SimilarityCondition", kind="smc"),
    )

    kind_registry.register(
        cls=BaseSampler,
        kind=SerializationKind(name="BaseSampler", kind="sm"),
    )
