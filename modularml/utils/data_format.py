from collections.abc import Sequence
from enum import Enum

import numpy as np

from modularml.utils.backend import Backend
from modularml.utils.optional_imports import check_tensorflow, check_torch


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


def format_requires_compatible_shapes(fmt: DataFormat) -> bool:
    """Returns True if the specified DataFormat requires compatible shapes."""
    return fmt in [DataFormat.NUMPY, DataFormat.TORCH, DataFormat.TENSORFLOW]


def format_is_tensorlike(fmt: DataFormat) -> bool:
    """True if specified DataFormat returns a tensor-like object."""
    fmt = normalize_format(fmt)
    return fmt in [DataFormat.NUMPY, DataFormat.TORCH, DataFormat.TENSORFLOW]


def infer_data_type(obj) -> str:
    torch = check_torch()
    tf = check_tensorflow()

    # Torch tensor
    if torch is not None and isinstance(obj, torch.Tensor):
        dt = obj.dtype
        if dt.is_floating_point:
            return "float"
        if dt in (torch.int8, torch.int16, torch.int32, torch.int64):
            return "int"
        if dt == torch.bool:
            return "bool"
        return "unknown"

    # TensorFlow tensor
    if tf is not None and isinstance(obj, tf.Tensor):
        dt = obj.dtype
        if dt.is_floating:
            return "float"
        if dt.is_integer:
            return "int"
        if dt.is_bool:
            return "bool"
        if dt == tf.string:
            return "string"
        return "unknown"

    # NumPy
    if isinstance(obj, np.ndarray) or np.isscalar(obj):
        if np.issubdtype(np.asarray(obj).dtype, np.floating):
            return "float"
        if np.issubdtype(np.asarray(obj).dtype, np.integer):
            return "int"
        if np.issubdtype(np.asarray(obj).dtype, np.bool_):
            return "bool"
        if np.issubdtype(np.asarray(obj).dtype, np.str_) or np.issubdtype(np.asarray(obj).dtype, np.object_):
            return "string"
        return "unknown"

    # Python primitives
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "string"

    return "unknown"


def get_data_format_for_backend(backend: str | Backend) -> DataFormat:
    if isinstance(backend, str):
        backend = Backend(backend)

    if backend == Backend.TORCH:
        return DataFormat.TORCH
    if backend == Backend.TENSORFLOW:
        return DataFormat.TENSORFLOW
    if backend in [Backend.SCIKIT, Backend.NONE]:
        return DataFormat.NUMPY
    msg = f"Unsupported backend: {backend}"
    raise ValueError(msg)


def ensure_list(x):
    """
    Ensure that the input is returned as a list.

    - None: return []
    - list: return itself (unchanged)
    - scalar (str, int, float, bool, etc.): return wrapped in a list
    - any other non-sequence type: return wrapped in a list
    - any other sequence (tuple, set, np.ndarray): return converted to list

    Args:
        x: Any input value.

    Returns:
        list: A list representation of `x`.

    Raises:
        TypeError: If the input is not convertible to a list.

    """
    if x is None:
        return []

    # If it's already a list, return directly
    if isinstance(x, list):
        return x

    # Treat strings and all scalar types as atomic -> wrap in list
    if isinstance(x, (str, bytes, int, float, bool)):
        return [x]

    # If it's a sequence (tuple, np.array, etc.), convert to list
    if isinstance(x, Sequence):
        return list(x)

    # For any other single object (e.g. Enum, custom class), also wrap in list
    return [x]
