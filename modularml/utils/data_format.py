



from typing import Any, Dict, Literal, Union
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


from modularml.utils.backend import Backend

from enum import Enum

class DataFormat(Enum):
    PANDAS = "pandas"
    NUMPY = "numpy"
    DICT = "dict"
    DICT_NUMPY = "dict_numpy"
    DICT_LIST = "dict_list"
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

def convert_to_format(data: Dict[str, Any], format: Union[str, DataFormat]) -> Any:
    fmt = normalize_format(format)
    
    if fmt == DataFormat.DICT:
        return data

    elif fmt == DataFormat.DICT_NUMPY:
        return {k: np.asarray(v) for k, v in data.items()}

    elif fmt == DataFormat.DICT_LIST:
        return {k: v.tolist() if isinstance(v, np.ndarray) else list(v) for k, v in data.items()}

    elif fmt == DataFormat.PANDAS:
        return pd.DataFrame(data)

    elif fmt == DataFormat.NUMPY:
        return np.column_stack(list(data.values()))

    elif fmt == DataFormat.LIST:
        return [list(x) for x in zip(*data.values())]

    elif fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return torch.from_numpy(np.column_stack(list(data.values()))).to(torch.float32)

    elif fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return tf.convert_to_tensor(np.column_stack(list(data.values())), dtype=tf.float32)

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
    
    

    