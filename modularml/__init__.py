try:
    from importlib.metadata import version

    __version__ = version("modularml")
except ImportError:
    __version__ = "unknown"


from .logger import logger
from .api import Backend, DataFormat, core, models

__all__ = [
    "Backend",
    "DataFormat",
    "core",
    "logger",
    "models",
]
