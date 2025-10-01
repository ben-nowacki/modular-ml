import logging
from pathlib import Path

FORMAT = "%(asctime)s.%(msecs)03d - [%(levelname)s] %(module)s.%(funcName)s(%(lineno)d): %(message)s"
LOG_FORMATTER = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S", fmt=FORMAT)


def set_logging_level(level):
    """
    Set the logging level for ModularML.

    Args:
        level (str): The logging level to set. Should be one of {'DEBUG', 'INFO', \
            'WARNING', 'ERROR', 'CRITICAL'}.

    """
    logger.setLevel(level)


def _create_logger(name: str, filename: str | Path | None = None):
    new_logger = logging.getLogger(name)
    handler = logging.StreamHandler() if filename is None else logging.FileHandler(filename)
    handler.setFormatter(LOG_FORMATTER)
    new_logger.addHandler(handler)
    return new_logger


logger = _create_logger(__name__)
set_logging_level("WARNING")
