from .core.splitting import registry as splitting_registry
from .core.transforms import registry as transform_registry

# from .core.sampling import registry as sampler_registry


def register_all():
    # Registers all splitters
    splitting_registry.register_classes()
    splitting_registry.register_kinds()

    # Registers all scalers
    transform_registry.register_classes()
    transform_registry.register_kinds()

    # # Registers all samplers
    # sampler_registry.register_classes()
    # sampler_registry.register_kinds()
