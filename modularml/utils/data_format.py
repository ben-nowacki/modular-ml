



from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple, Union
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


if TYPE_CHECKING:
    from modularml.core.data_structures.data import Data

from modularml.utils.backend import Backend

from enum import Enum


class DataFormat(Enum):
    PANDAS = "pandas"
    NUMPY = "numpy"
    DICT = "dict"
    DICT_NUMPY = "dict_numpy"
    DICT_LIST = "dict_list"
    DICT_TORCH = "dict_torch"
    DICT_TENSORFLOW = "dict_tensorflow"
    LIST = "list"
    TORCH = "torch.tensor"
    TENSORFLOW = "tensorflow.tensor"
    
_FORMAT_ALIASES = {
    "pandas": DataFormat.PANDAS,
    "pd": DataFormat.PANDAS,
    "df": DataFormat.PANDAS,

    "numpy": DataFormat.NUMPY,
    "np": DataFormat.NUMPY,

    "dict": DataFormat.DICT,
    "dict_numpy": DataFormat.DICT_NUMPY,
    "dict_list": DataFormat.DICT_LIST,
    "dict_torch": DataFormat.DICT_TORCH,
    "dict_tensorflow": DataFormat.DICT_TENSORFLOW,

    "list": DataFormat.LIST,

    "torch": DataFormat.TORCH,
    "torch.tensor": DataFormat.TORCH,

    "tf": DataFormat.TENSORFLOW,
    "tensorflow": DataFormat.TENSORFLOW,
    "tensorflow.tensor": DataFormat.TENSORFLOW,
}



def normalize_format(fmt: Union[str, DataFormat]) -> DataFormat:
    if isinstance(fmt, DataFormat):
        return fmt
    fmt = fmt.lower()
    if fmt not in _FORMAT_ALIASES:
        raise ValueError(f"Unknown data format: {fmt}")
    return _FORMAT_ALIASES[fmt]





T_ERRORS = Literal["raise", "coerce", "ignore"]

def to_list(obj: Any, errors: T_ERRORS = 'raise'):
    """
    Converts any object into a Python list.

    Args:
        obj: Any object to convert.
        errors: How to handle non-listable objects.
            - "raise": Raise TypeError if the object cannot be converted.
            - "coerce": Force conversion where possible (wrap scalars, arrays, tensors, etc.).
            - "ignore": Leave incompatible objects unchanged.

    Returns:
        list or object (if errors="ignore" and incompatible).
    """
    # If we're ignoring incompatible types, leave dicts unchanged directly
    if errors == "ignore" and isinstance(obj, dict):
        return obj


    py_obj = to_python(obj)
    
    # If it's already a list or tuple, convert directly
    if isinstance(py_obj, (list, tuple, np.ndarray)):
        return list(py_obj)
    
    # If it's a scalar, decide based on `errors`
    if np.isscalar(py_obj):
        return [py_obj]
    
    # Dicts aren't naturally convertible to lists
    if isinstance(py_obj, dict):
        if errors == "raise":
            raise TypeError(
                f"Cannot convert dict to list. Use DICT format instead."
            )
        elif errors == "coerce":
            # Convert dict values into a list of values
            return list(py_obj.values())
        elif errors == "ignore":
            return py_obj

    # Fallback: try NumPy coercion if possible
    try:
        return np.asarray(py_obj).tolist()
    except Exception:
        if errors == "raise":
            raise TypeError(
                f"Cannot convert object of type {type(py_obj)} to list."
            )
        elif errors == "ignore":
            return py_obj
        elif errors == "coerce":
            return [py_obj, ]
    
def to_python(obj):
    """
    Recursively converts an object into its native Python equivalent.

    Supported conversions:
    - NumPy scalars    -> Python scalars
    - NumPy arrays     -> Python lists
    - PyTorch tensors  -> Python scalars or lists
    - TensorFlow tensors -> Python scalars or lists
    - Dicts, tuples, and lists -> Recursively converted

    Args:
        obj: Any object to convert.

    Returns:
        Python-native object.
    """
    from modularml.core.data_structures.data import Data
    
    # NumPy
    if isinstance(obj, np.generic):           # np.int64, np.float64, etc.
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # PyTorch
    if torch is not None and isinstance(obj, torch.Tensor):
        # Move to CPU, detach from graph if needed, convert to list or scalar
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # TensorFlow
    if tf is not None and isinstance(obj, tf.Tensor):
        # Use .numpy() safely, then convert like numpy arrays
        np_obj = obj.numpy()
        if np_obj.ndim == 0:
            return np_obj.item()
        return np_obj.tolist()

    # Containers
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_python(v) for v in obj)
    elif isinstance(obj, Data):
        return to_python(obj.value)

    # Base case
    return obj

def to_numpy(obj: Any, errors: T_ERRORS = 'raise') -> np.ndarray:
    """
    Converts any object into a NumPy array.
    """
    # If it's already a numpy array, just return
    if isinstance(obj, np.ndarray):
        return obj
    
    py_obj = to_python(obj)

    # Dicts must use DICT_NUMPY format unless coerced
    if isinstance(py_obj, dict):
        if errors == "raise":
            raise TypeError("Cannot convert dict directly to NumPy array. Use DICT_NUMPY instead.")
        elif errors == "coerce":
            return np.array(list(py_obj.values()))
        elif errors == "ignore":
            return py_obj
    
    # Sequences (lists, tuples) -> convert directly
    if isinstance(py_obj, (list, tuple)):
        try:
            return np.asarray(py_obj)
        except Exception:
            if errors == "raise":
                raise TypeError(f"Cannot convert sequence of type {type(py_obj)} to NumPy array.")
            elif errors == "ignore":
                return py_obj
            elif errors == "coerce":
                return np.array([py_obj])

    # Scalars -> wrap into a 0-D array
    if np.isscalar(py_obj):
        return np.asarray(py_obj)
    
    # Unsupported type
    if errors == "raise":
        raise TypeError(f"Cannot convert object of type {type(py_obj)} to NumPy array.")
    elif errors == "ignore":
        return py_obj
    elif errors == "coerce":
        return np.array([py_obj])
    
def to_torch(obj: Any, errors: T_ERRORS = 'raise') -> "torch.Tensor": # type: ignore
    """
    Converts any object into a PyTorch tensor.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed.")

    py_obj = to_python(obj)
    try:
        return torch.as_tensor(np.asarray(py_obj), dtype=torch.float32)
    except Exception:
        if errors == "raise":
            raise TypeError(f"Cannot convert object of type {type(py_obj)} to Torch tensor.")
        elif errors == "ignore":
            return py_obj
        elif errors == "coerce":
            return torch.as_tensor(np.asarray([py_obj]), dtype=torch.float32)

def to_tensorflow(obj: Any, errors: T_ERRORS = 'raise') -> "tf.Tensor": # type: ignore
    """
    Converts any object into a TensorFlow tensor.
    """
    if tf is None:
        raise ImportError("TensorFlow is not installed.")

    py_obj = to_python(obj)
    try:
        return tf.convert_to_tensor(np.asarray(py_obj), dtype=tf.float32)
    except Exception:
        if errors == "raise":
            raise TypeError(f"Cannot convert object of type {type(py_obj)} to TensorFlow tensor.")
        elif errors == "ignore":
            return py_obj
        elif errors == "coerce":
            return tf.convert_to_tensor(np.asarray([py_obj]), dtype=tf.float32)
  
  
def format_has_shape(format: DataFormat) -> bool:
    """Returns True if the specified DataFormat has a shape attribute"""
    return format in [
        DataFormat.NUMPY, DataFormat.TORCH, DataFormat.TENSORFLOW
    ]
  
def enforce_numpy_shape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != target_shape:
        arr = arr.reshape(target_shape)
    return arr
  

def convert_dict_to_format(
    data: Dict[str, Any],
    format: Union[str, DataFormat],
    errors: T_ERRORS = 'raise',
) -> Any:
    """
    Converts a dictionary of data arrays into the specified format.

    Args:
        data: Dict of arrays, lists, scalars, or tensors.
        format: Target data format to convert into.
        errors: How to handle incompatible types:
            - "raise": Raise an error when conversion fails.
            - "coerce": Force conversion where possible.
            - "ignore": Leave unconvertible objects unchanged.

    Returns:
        Converted object.
    """
    fmt = normalize_format(format)

    if fmt == DataFormat.DICT:
        return to_python(data)

    elif fmt == DataFormat.DICT_LIST:
        # Force each value into a list or raise based on `errors`
        return {k: to_list(v, errors=errors) for k, v in data.items()}

    elif fmt == DataFormat.DICT_NUMPY:
        # Force each value into a numpy array or raise based on 'errors'
        return {k: to_numpy(v, errors=errors) for k, v in data.items()}
        
    elif fmt == DataFormat.DICT_TORCH:
        return {k: to_torch(v, errors=errors) for k, v in data.items()}
    
    elif fmt == DataFormat.DICT_TENSORFLOW:
        return {k: to_tensorflow(v, errors=errors) for k, v in data.items()}
        
    elif fmt == DataFormat.PANDAS:
        # Force each value into a list or raise based on 'errors'
        return pd.DataFrame({k: to_list(v, errors=errors) for k, v in data.items()})

    elif fmt == DataFormat.NUMPY:
        return np.column_stack([to_numpy(v, errors=errors) for v in data.values()])

    elif fmt == DataFormat.LIST:
        return [list(row) for row in zip(*[to_list(v, errors=errors) for v in data.values()])]

    elif fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return torch.tensor(
            np.column_stack([to_numpy(v, errors=errors) for v in data.values()]),
            dtype=torch.float32,
        )      
    elif fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return tf.convert_to_tensor(
            np.column_stack([to_numpy(v, errors=errors) for v in data.values()]),
            dtype=tf.float32,
        )

    else:
        raise ValueError(f"Unsupported data format: {fmt}")
    
def convert_to_format(
    data: Any,
    format: Union[str, DataFormat],
    errors: T_ERRORS = 'raise',
) -> Any:
    """
    Converts a data object into the specified format.

    Args:
        data: Dicts, arrays, lists, scalars, or tensors.
        format: Target data format to convert into.
        errors: How to handle incompatible types:
            - "raise": Raise an error when conversion fails.
            - "coerce": Force conversion where possible.
            - "ignore": Leave unconvertible objects unchanged.

    Returns:
        Converted object.
    """
    
    fmt = normalize_format(format)
    if isinstance(data, dict):
        return convert_dict_to_format(data=data, format=format, errors=errors)
    
    elif fmt == DataFormat.NUMPY:
        return to_numpy(data, errors=errors)
    
    elif fmt == DataFormat.LIST:
        return to_list(data, errors=errors)
    
    elif fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return to_torch(data, errors=errors)
         
    elif fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return to_tensorflow(data, errors=errors)
    
    else:
        raise ValueError(f"Unsupported data format: {fmt}")
    
def get_data_format_for_backend(backend: Union[str, Backend]) -> DataFormat:
    if isinstance(backend, str):
        backend = Backend(backend)
        
    if backend == Backend.TORCH:
        return DataFormat.TORCH
    elif backend == Backend.TENSORFLOW:
        return DataFormat.TENSORFLOW
    elif backend == Backend.SCIKIT:
        return DataFormat.NUMPY
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    

    