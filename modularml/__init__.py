

from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat



# Expose core and models as attributes for Sphinx autosummary
from . import core
from . import models

try:
    from importlib.metadata import version
    __version__ = version("ModularML")
except ImportError:
    __version__ = "unknown"