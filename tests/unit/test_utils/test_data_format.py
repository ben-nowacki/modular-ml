import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.data_format import (
    DataFormat,
    convert_dict_to_format,
    convert_to_format,
    enforce_numpy_shape,
    format_has_shape,
    get_data_format_for_backend,
    normalize_format,
    to_list,
    to_numpy,
    to_python,
    to_tensorflow,
    to_torch,
)
from modularml.utils.error_handling import ErrorMode


# ---------- normalize_format ----------
def test_normalize_format_aliases():
    assert normalize_format("pd") == DataFormat.PANDAS
    assert normalize_format("np") == DataFormat.NUMPY
    assert normalize_format("torch.tensor") == DataFormat.TORCH
    assert normalize_format(DataFormat.LIST) == DataFormat.LIST


def test_normalize_format_invalid():
    with pytest.raises(ValueError, match="Unknown data format: unknown"):
        normalize_format("unknown")


# ---------- to_python ----------
def test_to_python_numpy_and_pandas():
    arr = np.array([1, 2, 3])
    assert to_python(arr) == [1, 2, 3]

    scalar = np.int64(5)
    assert to_python(scalar) == 5

    series = pd.Series([1, 2])
    assert to_python(series) == [1, 2]


def test_to_python_torch_and_tf():
    t = torch.tensor([1.0, 2.0])
    assert to_python(t) == [1.0, 2.0]

    tf_t = tf.constant([3.0, 4.0])
    assert to_python(tf_t) == [3.0, 4.0]


def test_to_python_nested_dicts_lists():
    obj = {"a": np.array([1]), "b": [np.int64(2), 3]}
    result = to_python(obj)
    assert result == {"a": [1], "b": [2, 3]}


# ---------- to_list ----------
def test_to_list_scalars_and_arrays():
    assert to_list(5) == [5]
    assert to_list(np.array([1, 2])) == [1, 2]


def test_to_list_dict_behavior():
    d = {"a": 1, "b": 2}
    with pytest.raises(TypeError):
        to_list(d, errors=ErrorMode.RAISE)
    assert to_list(d, errors=ErrorMode.COERCE) == [1, 2]
    assert to_list(d, errors=ErrorMode.IGNORE) == d


# ---------- to_numpy ----------
def test_to_numpy_scalars_and_arrays():
    arr = to_numpy(5)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == ()

    arr2 = to_numpy([1, 2, 3])
    np.testing.assert_array_equal(arr2, np.array([1, 2, 3]))


def test_to_numpy_dict_behavior():
    d = {"a": 1}
    with pytest.raises(TypeError):
        to_numpy(d, errors=ErrorMode.RAISE)
    arr = to_numpy(d, errors=ErrorMode.COERCE)
    assert isinstance(arr, np.ndarray)


# ---------- Torch / TensorFlow ----------
def test_to_torch_and_tensorflow():
    arr = [1, 2, 3]
    t = to_torch(arr)
    assert isinstance(t, torch.Tensor)
    tf_t = to_tensorflow(arr)
    assert isinstance(tf_t, tf.Tensor)


# ---------- convert_dict_to_format ----------
def test_convert_dict_to_format_numpy_and_list():
    d = {"x": [1, 2], "y": [3, 4]}
    np_arr = convert_dict_to_format(d, "numpy")
    assert isinstance(np_arr, np.ndarray)
    list_out = convert_dict_to_format(d, "list")
    assert all(isinstance(row, list) for row in list_out)


def test_convert_dict_to_format_torch_and_tf():
    d = {"x": [1, 2], "y": [3, 4]}
    t = convert_dict_to_format(d, "dict_torch")
    assert all(isinstance(v, torch.Tensor) for v in t.values())
    tf_d = convert_dict_to_format(d, "dict_tensorflow")
    assert all(isinstance(v, tf.Tensor) for v in tf_d.values())


# ---------- convert_to_format ----------
def test_convert_to_format_numpy_and_list():
    arr = [1, 2, 3]
    np_arr = convert_to_format(arr, "numpy")
    assert isinstance(np_arr, np.ndarray)
    lst = convert_to_format(arr, "list")
    assert lst == [1, 2, 3]


def test_convert_to_format_invalid():
    with pytest.raises(ValueError, match="Unknown data format: invalid"):
        convert_to_format([1], "invalid")


# ---------- format_has_shape / enforce_numpy_shape ----------
def test_format_has_shape_and_enforce_shape():
    assert format_has_shape(DataFormat.NUMPY)
    assert not format_has_shape(DataFormat.DICT)
    arr = np.array([1, 2, 3, 4])
    reshaped = enforce_numpy_shape(arr, (2, 2))
    assert reshaped.shape == (2, 2)


# ---------- get_data_format_for_backend ----------
def test_get_data_format_for_backend():
    assert get_data_format_for_backend(Backend.TORCH) == DataFormat.TORCH
    assert get_data_format_for_backend("torch") == DataFormat.TORCH
    assert get_data_format_for_backend(Backend.TENSORFLOW) == DataFormat.TENSORFLOW
    assert get_data_format_for_backend("tensorflow") == DataFormat.TENSORFLOW
    assert get_data_format_for_backend(Backend.SCIKIT) == DataFormat.NUMPY
    assert get_data_format_for_backend("scikit") == DataFormat.NUMPY
    assert get_data_format_for_backend(Backend.NONE) == DataFormat.NUMPY
    assert get_data_format_for_backend("none") == DataFormat.NUMPY

    with pytest.raises(ValueError, match=r"'some_other_backend' is not a valid Backend"):
        get_data_format_for_backend("some_other_backend")
