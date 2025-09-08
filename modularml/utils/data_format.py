from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.error_handling import ErrorMode


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


def normalize_format(fmt: str | DataFormat) -> DataFormat:
    if isinstance(fmt, DataFormat):
        return fmt
    fmt = fmt.lower()
    if fmt not in _FORMAT_ALIASES:
        msg = f"Unknown data format: {fmt}"
        raise ValueError(msg)
    return _FORMAT_ALIASES[fmt]


def to_python(obj):  # noqa: PLR0911
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
    # NumPy
    if isinstance(obj, np.generic):  # np.int64, np.float64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas
    if isinstance(obj, pd.Series):
        return to_python(obj.values)

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
    if isinstance(obj, list | tuple):
        return type(obj)(to_python(v) for v in obj)
    if hasattr(obj, "value"):
        return to_python(obj.value)

    # Base case
    return obj


def to_list(obj: Any, errors: ErrorMode = ErrorMode.RAISE):  # noqa: PLR0911
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
    if errors == ErrorMode.IGNORE and isinstance(obj, dict):
        return obj

    py_obj = to_python(obj)

    # If it's already a list or tuple, convert directly
    if isinstance(py_obj, list | tuple | np.ndarray):
        return list(py_obj)

    # If it's a scalar, decide based on `errors`
    if np.isscalar(py_obj):
        return [py_obj]

    # Dicts aren't naturally convertible to lists
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError("Cannot convert dict to list. Use DICT format instead.")
        if errors == ErrorMode.COERCE:
            # Convert dict values into a list of values
            return list(py_obj.values())
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Fallback: try NumPy coercion if possible
    try:
        return np.asarray(py_obj).tolist()
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to list."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return [py_obj]


def to_numpy(obj: Any, errors: ErrorMode = ErrorMode.RAISE) -> np.ndarray:  # noqa: PLR0911
    """Converts any object into a NumPy array."""
    # If it's already a numpy array, just return
    if isinstance(obj, np.ndarray):
        return obj

    py_obj = to_python(obj)

    # Dicts must use DICT_NUMPY format unless coerced
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError("Cannot convert dict directly to NumPy array. Use DICT_NUMPY instead.")
        if errors == ErrorMode.COERCE:
            return np.array(list(py_obj.values()))
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Sequences (lists, tuples) -> convert directly
    if isinstance(py_obj, list | tuple):
        try:
            return np.asarray(py_obj)

        except Exception as e:
            if errors == ErrorMode.RAISE:
                msg = f"Cannot convert sequence of type {type(py_obj)} to NumPy array."
                raise TypeError(msg) from e
            if errors == ErrorMode.IGNORE:
                return py_obj
            if errors == ErrorMode.COERCE:
                return np.array([py_obj])

    # Scalars -> wrap into a 0-D array
    if np.isscalar(py_obj):
        return np.asarray(py_obj)

    # Unsupported type
    if errors == ErrorMode.RAISE:
        msg = f"Cannot convert object of type {type(py_obj)} to NumPy array."
        raise TypeError(msg)
    if errors == ErrorMode.IGNORE:
        return py_obj
    if errors == ErrorMode.COERCE:
        return np.array([py_obj])
    return None


def to_torch(obj: Any, errors: ErrorMode = ErrorMode.RAISE) -> "torch.Tensor":
    """Converts any object into a PyTorch tensor."""
    if torch is None:
        raise ImportError("PyTorch is not installed.")

    # If it's already a Torch Tensor, just return
    if isinstance(obj, torch.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return torch.as_tensor(np.asarray(py_obj), dtype=torch.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to Torch tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return torch.as_tensor(np.asarray([py_obj]), dtype=torch.float32)


def to_tensorflow(obj: Any, errors: ErrorMode = ErrorMode.RAISE) -> "tf.Tensor":
    """Converts any object into a TensorFlow tensor."""
    if tf is None:
        raise ImportError("TensorFlow is not installed.")

    # If it's already a Tensforflow Tensor, just return
    if isinstance(obj, tf.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return tf.convert_to_tensor(np.asarray(py_obj), dtype=tf.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to TensorFlow tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return tf.convert_to_tensor(np.asarray([py_obj]), dtype=tf.float32)


def format_has_shape(fmt: DataFormat) -> bool:
    """Returns True if the specified DataFormat has a shape attribute."""
    return fmt in [DataFormat.NUMPY, DataFormat.TORCH, DataFormat.TENSORFLOW]


def enforce_numpy_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != target_shape:
        arr = arr.reshape(target_shape)
    return arr


def convert_dict_to_format(  # noqa: PLR0911
    data: dict[str, Any],
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
) -> Any:
    """
    Converts a dictionary of data arrays into the specified format.

    Args:
        data: Dict of arrays, lists, scalars, or tensors.
        fmt: Target data format to convert into.
        errors: How to handle incompatible types:
            - ErrorMode.RAISE: Raise an error when conversion fails.
            - ErrorMode.COERCE: Force conversion where possible.
            - ErrorMode.IGNORE: Leave unconvertible objects unchanged.

    Returns:
        Converted object.

    """
    fmt = normalize_format(fmt)

    if fmt == DataFormat.DICT:
        return to_python(data)

    if fmt == DataFormat.DICT_LIST:
        # Force each value into a list or raise based on `errors`
        return {k: to_list(v, errors=errors) for k, v in data.items()}

    if fmt == DataFormat.DICT_NUMPY:
        # Force each value into a numpy array or raise based on 'errors'
        return {k: to_numpy(v, errors=errors) for k, v in data.items()}

    if fmt == DataFormat.DICT_TORCH:
        return {k: to_torch(v, errors=errors) for k, v in data.items()}

    if fmt == DataFormat.DICT_TENSORFLOW:
        return {k: to_tensorflow(v, errors=errors) for k, v in data.items()}

    if fmt == DataFormat.PANDAS:
        # Force each value into a list or raise based on 'errors'
        return pd.DataFrame({k: to_list(v, errors=errors) for k, v in data.items()})

    if fmt == DataFormat.NUMPY:
        return np.column_stack([to_numpy(v, errors=errors) for v in data.values()])

    if fmt == DataFormat.LIST:
        return [list(row) for row in zip(*[to_list(v, errors=errors) for v in data.values()], strict=True)]

    if fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return torch.tensor(
            np.column_stack([to_numpy(v, errors=errors) for v in data.values()]),
            dtype=torch.float32,
        )
    if fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return tf.convert_to_tensor(
            np.column_stack([to_numpy(v, errors=errors) for v in data.values()]),
            dtype=tf.float32,
        )

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)


def convert_to_format(
    data: Any,
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
) -> Any:
    """
    Converts a data object into the specified format.

    Args:
        data: Dicts, arrays, lists, scalars, or tensors.
        fmt: Target data format to convert into.
        errors: How to handle incompatible types:
            - ErrorMode.RAISE: Raise an error when conversion fails.
            - ErrorMode.COERCE: Force conversion where possible.
            - ErrorMode.IGNORE: Leave unconvertible objects unchanged.

    Returns:
        Converted object.

    """
    fmt = normalize_format(fmt)
    if isinstance(data, dict):
        return convert_dict_to_format(data=data, fmt=fmt, errors=errors)

    if fmt == DataFormat.NUMPY:
        return to_numpy(data, errors=errors)

    if fmt == DataFormat.LIST:
        return to_list(data, errors=errors)

    if fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return to_torch(data, errors=errors)

    if fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return to_tensorflow(data, errors=errors)

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)


def get_data_format_for_backend(backend: str | Backend) -> DataFormat:
    if isinstance(backend, str):
        backend = Backend(backend)

    if backend == Backend.TORCH:
        return DataFormat.TORCH
    if backend == Backend.TENSORFLOW:
        return DataFormat.TENSORFLOW
    if backend == Backend.SCIKIT:
        return DataFormat.NUMPY
    msg = f"Unsupported backend: {backend}"
    raise ValueError(msg)
