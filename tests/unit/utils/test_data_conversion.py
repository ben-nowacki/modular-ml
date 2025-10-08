import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.data_conversion import (
    align_ranks,
    convert_dict_to_format,
    convert_to_format,
    enforce_numpy_shape,
    to_list,
    to_numpy,
    to_python,
    to_tensorflow,
    to_torch,
)
from modularml.utils.error_handling import ErrorMode


# ---------- to_python ----------
@pytest.mark.unit
def test_to_python_numpy_pandas_torch_tf():
    arr = np.array([1, 2, 3])
    assert to_python(arr) == [1, 2, 3]
    assert to_python(np.int64(5)) == 5

    s = pd.Series([1, 2])
    assert to_python(s) == [1, 2]

    t = torch.tensor([1.0, 2.0])
    assert to_python(t) == [1.0, 2.0]

    tf_t = tf.constant([3.0, 4.0])
    assert to_python(tf_t) == [3.0, 4.0]


@pytest.mark.unit
def test_to_python_nested_structures():
    obj = {"a": np.array([1]), "b": [np.int64(2), 3]}
    result = to_python(obj)
    assert result == {"a": [1], "b": [2, 3]}


# ---------- to_list ----------
@pytest.mark.unit
def test_to_list_scalars_arrays_and_dicts():
    assert to_list(5) == [5]
    assert to_list(np.array([1, 2])) == [1, 2]

    d = {"a": 1, "b": 2}
    with pytest.raises(TypeError):
        to_list(d, errors=ErrorMode.RAISE)
    assert to_list(d, errors=ErrorMode.COERCE) == [1, 2]
    assert to_list(d, errors=ErrorMode.IGNORE) == d


# ---------- to_numpy ----------
@pytest.mark.unit
def test_to_numpy_scalars_and_lists():
    arr = to_numpy(5)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == ()

    arr2 = to_numpy([1, 2, 3])
    np.testing.assert_array_equal(arr2, np.array([1, 2, 3]))


@pytest.mark.unit
def test_to_numpy_dict_behavior():
    d = {"a": 1}
    with pytest.raises(TypeError):
        to_numpy(d, errors=ErrorMode.RAISE)
    coerced = to_numpy(d, errors=ErrorMode.COERCE)
    assert isinstance(coerced, np.ndarray)


# ---------- Torch / TensorFlow ----------
@pytest.mark.unit
def test_to_torch_and_tensorflow_conversion():
    arr = [1, 2, 3]
    t = to_torch(arr)
    assert isinstance(t, torch.Tensor)
    tf_t = to_tensorflow(arr)
    assert isinstance(tf_t, tf.Tensor)


# ---------- convert_dict_to_format ----------
@pytest.mark.unit
def test_convert_dict_to_format_variants():
    d = {"x": [1, 2], "y": [3, 4]}
    np_arr = convert_dict_to_format(d, "numpy")
    assert isinstance(np_arr, np.ndarray)

    list_out = convert_dict_to_format(d, "list")
    assert all(isinstance(row, list) for row in list_out)

    t_dict = convert_dict_to_format(d, "dict_torch")
    assert all(isinstance(v, torch.Tensor) for v in t_dict.values())

    tf_dict = convert_dict_to_format(d, "dict_tensorflow")
    assert all(isinstance(v, tf.Tensor) for v in tf_dict.values())


# ---------- convert_to_format ----------
@pytest.mark.unit
def test_convert_to_format_numpy_and_list():
    arr = [1, 2, 3]
    np_arr = convert_to_format(arr, "numpy")
    assert isinstance(np_arr, np.ndarray)
    lst = convert_to_format(arr, "list")
    assert lst == [1, 2, 3]


@pytest.mark.unit
def test_convert_to_format_invalid_format():
    with pytest.raises(ValueError, match="Unknown data format: invalid"):
        convert_to_format([1], "invalid")


# ---------- enforce_numpy_shape ----------
@pytest.mark.unit
def test_enforce_numpy_shape():
    arr = np.arange(4)
    reshaped = enforce_numpy_shape(arr, (2, 2))
    assert reshaped.shape == (2, 2)


# ---------- align_ranks ----------
@pytest.mark.unit
def test_align_ranks_numpy_simple():
    a = np.zeros((32, 1, 4))
    b = np.zeros((32, 4))
    a2, b2 = align_ranks(a, b, Backend.SCIKIT)
    assert a2.shape == b2.shape == (32, 1, 4)


@pytest.mark.unit
def test_align_ranks_numpy_multi_missing_dims():
    a = np.zeros((32, 1, 1, 4))
    b = np.zeros((32, 4))
    a2, b2 = align_ranks(a, b, Backend.SCIKIT)
    assert a2.shape == b2.shape == (32, 1, 1, 4)


@pytest.mark.unit
def test_align_ranks_same_rank_different_shapes_raises():
    a = np.zeros((32, 1, 4))
    b = np.zeros((32, 4, 1))
    with pytest.raises(ValueError, match="same rank but not in the same order"):
        align_ranks(a, b, Backend.SCIKIT)


@pytest.mark.unit
def test_align_ranks_incompatible_shape_raises():
    a = np.zeros((32, 1, 4))
    b = np.zeros((32, 1))
    with pytest.raises(ValueError, match="cannot be aligned"):
        align_ranks(a, b, Backend.SCIKIT)


@pytest.mark.unit
def test_align_ranks_torch_and_tf():
    t = torch.zeros((32, 1, 4))
    t2 = torch.zeros((32, 4))
    a, b = align_ranks(t, t2, Backend.TORCH)
    assert a.shape == b.shape == (32, 1, 4)

    tf_a = tf.zeros((32, 1, 4))
    tf_b = tf.zeros((32, 4))
    a, b = align_ranks(tf_a, tf_b, Backend.TENSORFLOW)
    assert a.shape == b.shape == (32, 1, 4)
