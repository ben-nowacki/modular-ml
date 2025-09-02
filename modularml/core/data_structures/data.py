
from enum import Enum
import numpy as np
import torch
import tensorflow as tf
from typing import Any, Union

from modularml.utils.backend import Backend


class Data:
    """A container to wrap any backend-specific data type"""
    
    def __init__(self, value: Any):
        self.value = value
        self._inferred_backend = self._infer_backend()
        
        
    def _infer_backend(self) -> Backend:
        if isinstance(self.value, torch.Tensor):
            return Backend.TORCH
        elif isinstance(self.value, tf.Tensor):
            return Backend.TENSORFLOW
        elif isinstance(self.value, (np.ndarray, np.generic)):
            return Backend.SCIKIT
        elif isinstance(self.value, (int, float, list, bool)):
            return Backend.NONE
        else:
            raise TypeError(f"Unsupported type for Data: {type(self.value)}")

    @property
    def backend(self) -> Backend:
        return self._inferred_backend

    @property
    def shape(self):
        if hasattr(self.value, 'shape'):
            return self.value.shape
        else:
            return tuple(np.asarray(self.value).shape)

    @property
    def dtype(self):
        if hasattr(self.value, 'dtype'):
            return self.value.dtype
        else: 
            return type(self.value)
    
    def __len__(self):
        try:
            return len(self.value)
        except TypeError:
            raise TypeError(f"Data object wrapping {type(self.value)} has no length")
    
    def __getitem__(self, key) -> "Data":
        return Data(self.value[key])
    
    def __repr__(self):
        return f"Data(backend={self.backend}, shape={self.shape}, dtype={self.dtype})"
    
    def __eq__(self, other):
        if isinstance(other, Data):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self,):
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
        elif self.backend == Backend.TENSORFLOW:
            return self.value.numpy().astype(dtype)
        elif self.backend in (Backend.SCIKIT, Backend.NONE):
            return np.array(self.value, dtype=dtype)
        else:
            raise RuntimeError("Cannot convert unknown backend to NumPy")

    def to_torch(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.backend == Backend.TORCH:
            return self.value.to(dtype=dtype)
        return torch.from_numpy(self.to_numpy()).to(dtype=dtype)

    def to_tensorflow(self, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
        if self.backend == Backend.TENSORFLOW:
            return tf.cast(self.value, dtype)
        return tf.convert_to_tensor(self.to_numpy(), dtype=dtype)

    def to_backend(self, target: Union[str, Backend]) -> Union[np.ndarray, torch.Tensor, tf.Tensor]:
        if isinstance(target, str):
            target = Backend(target)
        if target == Backend.TORCH:
            return self.to_torch()
        elif target == Backend.TENSORFLOW:
            return self.to_tensorflow()
        elif target == Backend.SCIKIT:
            return self.to_numpy()
        else:
            raise ValueError(f"Unsupported target backend: {target}")

    # ==================================================================
    # Data-Wrapped Conversions
    # ==================================================================
    def as_numpy(self, dtype: np.dtype = np.float32) -> "Data":
        return Data(self.to_numpy(dtype))

    def as_torch(self, dtype: torch.dtype = torch.float32) -> "Data":
        return Data(self.to_torch(dtype))

    def as_tensorflow(self, dtype: tf.dtypes.DType = tf.float32) -> "Data":
        return Data(self.to_tensorflow(dtype))

    def as_backend(self, target: Union[str, Backend]) -> "Data":
        return Data(self.to_backend(target))
   