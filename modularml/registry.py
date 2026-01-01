from .core.splitting import registry as splitting_registry
from .core.transforms import registry as transform_registry

from .core.sampling import registry as sampler_registry
from .core.data import registry as data_registry
from .core.models import registry as model_registry


def register_all():
    # Registers all splitters
    splitting_registry.register_builtin()
    splitting_registry.register_kinds()

    # Registers all scalers
    transform_registry.register_builtin()
    transform_registry.register_kinds()

    # Registers all samplers
    sampler_registry.register_builtin()
    sampler_registry.register_kinds()

    # Register FeatureSet
    data_registry.register_builtin()
    data_registry.register_kinds()

    # Register base models
    model_registry.register_builtin()
    model_registry.register_kinds()
