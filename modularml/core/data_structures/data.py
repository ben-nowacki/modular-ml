from typing import Any, Union

import numpy as np
import tensorflow as tf
import torch

from modularml.utils.backend import Backend


class Data:
    """
    A backend-agnostic wrapper around a data value (e.g., tensor or array).

    Supports native types from PyTorch, TensorFlow, NumPy, and Python primitives,
    and provides utility methods for conversion between formats.
    """

    def __init__(self, value: Any):
        """
        Initialize a new Data object.

        Args:
            value (Any): The value to wrap. Can be a tensor (Torch, TF), NumPy array,
                or a primitive Python value (int, float, list, bool).

        """
        self.value = value
        self._inferred_backend = self._infer_backend()

    def _infer_backend(self) -> Backend:
        """
        Infer the backend of the wrapped data.

        Returns:
            Backend: The detected backend.

        """
        if isinstance(self.value, torch.Tensor):
            return Backend.TORCH
        if isinstance(self.value, tf.Tensor):
            return Backend.TENSORFLOW
        if isinstance(self.value, np.ndarray | np.generic):
            return Backend.SCIKIT
        if isinstance(self.value, int | float | list | bool):
            return Backend.NONE
        msg = f"Unsupported type for Data: {type(self.value)}"
        raise TypeError(msg)

    @property
    def backend(self) -> Backend:
        """
        The backend associated with the wrapped value.

        Returns:
            Backend: One of TORCH, TENSORFLOW, SCIKIT, or NONE.

        """
        return self._inferred_backend

    @property
    def shape(self):
        """
        Shape of the wrapped value.

        Returns:
            tuple: The shape of the data as a tuple of dimensions.

        """
        if hasattr(self.value, "shape"):
            return self.value.shape
        return tuple(np.asarray(self.value).shape)

    @property
    def dtype(self):
        """
        Data type of the wrapped value.

        Returns:
            Any: The data type (e.g., torch.float32, np.float64).

        """
        if hasattr(self.value, "dtype"):
            return self.value.dtype
        return type(self.value)

    def __len__(self):
        """
        Return the length of the wrapped data.

        Raises:
            TypeError: If the wrapped object has no defined length.

        Returns:
            int: The length of the data.

        """
        try:
            return len(self.value)
        except TypeError as e:
            msg = f"Data object wrapping {type(self.value)} has no length"
            raise TypeError(msg) from e

    def __getitem__(self, key) -> "Data":
        """
        Get a sliced subset of the wrapped data.

        Args:
            key (Any): A valid index or slice.

        Returns:
            Data: A new Data object containing the sliced value.

        """
        if hasattr(self.value, "__getitem__"):
            return Data(self.value[key])
        raise AttributeError("Data value does not support __getitem__.")

    def __repr__(self):
        return f"Data(backend={self.backend}, shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other):
        if isinstance(other, Data):
            return self.value == other.value
        return self.value == other

    def __hash__(
        self,
    ):
        return hash(self.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.value < (other.value if isinstance(other, Data) else other)

    def __le__(self, other):
        return self.value <= (other.value if isinstance(other, Data) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other, Data) else other)

    def __ge__(self, other):
        return self.value >= (other.value if isinstance(other, Data) else other)

    # ==================================================================
    # Raw Conversions
    # ==================================================================
    def to_numpy(self, dtype: np.dtype = np.float32) -> np.ndarray:
        if self.backend == Backend.TORCH:
            return self.value.detach().cpu().numpy().astype(dtype)
        if self.backend == Backend.TENSORFLOW:
            return self.value.numpy().astype(dtype)
        if self.backend in (Backend.SCIKIT, Backend.NONE):
            return np.array(self.value, dtype=dtype)
        raise RuntimeError("Cannot convert unknown backend to NumPy")

    def to_torch(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.backend == Backend.TORCH:
            return self.value.to(dtype=dtype)
        return torch.from_numpy(self.to_numpy()).to(dtype=dtype)

    def to_tensorflow(self, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
        if self.backend == Backend.TENSORFLOW:
            return tf.cast(self.value, dtype)
        return tf.convert_to_tensor(self.to_numpy(), dtype=dtype)

    def to_backend(self, target: str | Backend) -> np.ndarray | torch.Tensor | tf.Tensor:
        if isinstance(target, str):
            target = Backend(target)
        if target == Backend.TORCH:
            return self.to_torch()
        if target == Backend.TENSORFLOW:
            return self.to_tensorflow()
        if target == Backend.SCIKIT:
            return self.to_numpy()
        msg = f"Unsupported target backend: {target}"
        raise ValueError(msg)

    # ==================================================================
    # Data-Wrapped Conversions
    # ==================================================================
    def as_numpy(self, dtype: np.dtype = np.float32) -> "Data":
        return Data(self.to_numpy(dtype))

    def as_torch(self, dtype: torch.dtype = torch.float32) -> "Data":
        return Data(self.to_torch(dtype))

    def as_tensorflow(self, dtype: tf.dtypes.DType = tf.float32) -> "Data":
        return Data(self.to_tensorflow(dtype))

    def as_backend(self, target: str | Backend) -> "Data":
        return Data(self.to_backend(target))
