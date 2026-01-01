from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    """
    Import and return a Python object from a fully-qualified import path.

    Examples:
        >>> import_object("torch.nn.Linear")
        >>> import_object("modularml.models.MODEL_REGISTRY")

    Args:
        path (str):
            Fully-qualified import path to a module attribute.

    Returns:
        Any:
            The imported object.

    Raises:
        ImportError:
            If the module cannot be imported.
        AttributeError:
            If the attribute does not exist in the module.
        ValueError:
            If the path is malformed.

    """
    if not isinstance(path, str) or "." not in path:
        msg = f"Invalid import path: {path!r}"
        raise ValueError(msg)

    module_path, attr_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        msg = f"Failed to import module '{module_path}' while resolving '{path}'"
        raise ImportError(msg) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        msg = f"Module '{module_path}' has no attribute '{attr_name}' while resolving '{path}'"
        raise AttributeError(msg) from exc
