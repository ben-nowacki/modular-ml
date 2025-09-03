
from enum import Enum

class Backend(str, Enum):
    TORCH = 'torch'
    TENSORFLOW = 'tensorflow'
    SCIKIT = 'scikit'
    NONE = 'none'


def backend_requires_optimizer(backend: Backend) -> bool:
    return backend in [
        Backend.TORCH, Backend.TENSORFLOW
    ]