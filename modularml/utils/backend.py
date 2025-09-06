from enum import Enum
from typing import Any


class Backend(str, Enum):
    """
    Enum representing supported model backends in ModularML.

    Attributes:
        TORCH (str): PyTorch backend (torch.nn.Module, torch.Tensor).
        TENSORFLOW (str): TensorFlow backend (tf.keras.Model, tf.Tensor).
        SCIKIT (str): scikit-learn backend (BaseEstimator).
        NONE (str): No backend detected or unsupported object type.
    """

    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    SCIKIT = "scikit"
    NONE = "none"


def backend_requires_optimizer(backend: Backend) -> bool:
    """
    Determines whether a given backend requires an optimizer for training.

    Args:
        backend (Backend): The backend to check.

    Returns:
        bool: True if the backend requires an optimizer (e.g., PyTorch, TensorFlow),
              False otherwise (e.g., scikit-learn).
    """
    return backend in [Backend.TORCH, Backend.TENSORFLOW]


def infer_backend(obj: Any) -> Backend:
    """
    Infers the model backend associated with a given object.

    Supports inference from:
    - Data objects (e.g., torch.Tensor, tf.Tensor)
    - Model instances (e.g., torch.nn.Module, tf.keras.Model, sklearn.BaseEstimator)
    - Model classes (e.g., torch.nn.Module subclasses)

    Args:
        obj (Any): The object to inspect.

    Returns:
        Backend: The inferred backend enum value. Returns Backend.NONE if no known backend is detected.

    """
    try:
        import torch  # noqa: PLC0415

        if isinstance(obj, torch.Tensor | torch.nn.Module) or (
            isinstance(obj, type) and issubclass(obj, torch.nn.Module)
        ):
            return Backend.TORCH
    except ImportError:
        pass

    try:
        import tensorflow as tf  # noqa: PLC0415

        if isinstance(obj, tf.Tensor | tf.keras.Model) or (isinstance(obj, type) and issubclass(obj, tf.keras.Model)):
            return Backend.TENSORFLOW
    except ImportError:
        pass

    try:
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        if isinstance(obj, BaseEstimator) or (isinstance(obj, type) and issubclass(obj, BaseEstimator)):
            return Backend.SCIKIT
    except ImportError:
        pass

    return Backend.NONE
