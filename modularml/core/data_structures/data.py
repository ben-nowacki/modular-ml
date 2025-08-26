
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
        elif isinstance(self.value, (int, float, bool)):
            return Backend.SCIKIT
        else:
            raise TypeError(f"Unsupported type for Data: {type(self.value)}")

    @property
    def backend(self) -> Backend:
        return self._inferred_backend

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype
        
    def __getitem__(self, key) -> "Data":
        return Data(self.value[key])
    
    def __repr__(self):
        return f"Data(backend={self.backend}, shape={self.shape}, dtype={self.dtype})"
    
    def __eq__(self, other):
        if isinstance(other, Data):
            return self.value == other.value
        return self.value == other
    
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
    

    def to_numpy(self) -> np.ndarray:
        if self.backend == Backend.TORCH:
            return self.value.detach().cpu().numpy()
        elif self.backend == Backend.TENSORFLOW:
            return self.value.numpy()
        elif self.backend == Backend.SCIKIT:
            return self.value
        else:
            raise RuntimeError(f"Cannot convert unknown backend to NumPy")

    def to_torch(self) -> torch.Tensor:
        if self.backend == Backend.TORCH:
            return self.value
        else:
            return torch.from_numpy(self.to_numpy())

    def to_tensorflow(self) -> tf.Tensor:
        if self.backend == Backend.TENSORFLOW:
            return self.value
        else:
            return tf.convert_to_tensor(self.to_numpy())

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


    