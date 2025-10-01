from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat

from .logger import logger

# Expose core and models as attributes for Sphinx autosummary
from . import core, models

try:
    from importlib.metadata import version

    __version__ = version("modularml")
except ImportError:
    __version__ = "unknown"


__all__ = [
    "Backend",
    "DataFormat",
    "core",
    "logger",
    "models",
]
