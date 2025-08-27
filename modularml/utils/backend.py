
from enum import Enum

class Backend(str, Enum):
    TORCH = 'torch'
    TENSORFLOW = 'tensorflow'
    SCIKIT = 'scikit'
    NONE = 'none'

