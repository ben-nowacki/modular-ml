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

    def __repr__(self) -> str:
        return self.value


def normalize_backend(value: str | Backend):
    if isinstance(value, Backend):
        return value
    if isinstance(value, str):
        value = value.lower().strip("backend").strip(".")
        if value in ["torch", "pytorch"]:
            return Backend.TORCH
        if value in ["tensorflow", "tf", "keras"]:
            return Backend.TENSORFLOW
        if value in ["scikit", "sklearn"]:
            return Backend.SCIKIT
        return Backend(value)
    msg = f"Unsupported Backend value: {value}"
    raise ValueError(msg)


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


def infer_backend(obj_or_cls: Any) -> Backend:
    """
    Infers the backend associated with a given object or class.

    Supports inference from:
    - Data objects (e.g., torch.Tensor, tf.Tensor)
    - Model instances and classes
    - Optimizer instances and classes

    Args:
        obj_or_cls (Any): The object/class to inspect.

    Returns:
        Backend: The inferred backend enum value.

    """

    def _safe_issubclass(obj_or_cls, bases):
        try:
            return isinstance(obj_or_cls, type) and issubclass(obj_or_cls, bases)
        except TypeError:
            return False

    # ================================================
    # PyTorch
    # ================================================
    try:
        import torch

        if isinstance(obj_or_cls, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer)) or (
            isinstance(obj_or_cls, type) and _safe_issubclass(obj_or_cls, (torch.nn.Module, torch.optim.Optimizer))
        ):
            return Backend.TORCH
    except ImportError:
        pass

    # ================================================
    # Tensorflow
    # ================================================
    try:
        import tensorflow as tf

        if isinstance(obj_or_cls, (tf.Tensor, tf.keras.Model, tf.keras.losses.Loss, tf.keras.optimizers.Optimizer)) or (
            isinstance(obj_or_cls, type)
            and _safe_issubclass(obj_or_cls, (tf.keras.Model, tf.keras.losses.Loss, tf.keras.optimizers.Optimizer))
        ):
            return Backend.TENSORFLOW
    except ImportError:
        pass

    # ================================================
    # Sklearn / Numpy
    # ================================================
    try:
        from sklearn.base import BaseEstimator

        if isinstance(obj_or_cls, BaseEstimator) or (
            isinstance(obj_or_cls, type) and _safe_issubclass(obj_or_cls, BaseEstimator)
        ):
            return Backend.SCIKIT
    except ImportError:
        pass

    return Backend.NONE
