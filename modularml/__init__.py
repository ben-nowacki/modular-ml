

from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat



try:
    from importlib.metadata import version
    __version__ = version("ModularML")
except ImportError:
    __version__ = "unknown"