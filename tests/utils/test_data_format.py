import pytest
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

from modularml.utils.data_format import (
    DataFormat,
    normalize_format,
    to_list,
    to_python,
    to_numpy,
    to_torch,
    to_tensorflow,
    convert_dict_to_format,
    get_data_format_for_backend,
)
from modularml.utils.backend import Backend


# ----------------------------------------------------
# Tests for normalize_format
# ----------------------------------------------------

def test_normalize_format_valid_aliases():
    assert normalize_format("pd") == DataFormat.PANDAS
    assert normalize_format("numpy") == DataFormat.NUMPY
    assert normalize_format("dict_list") == DataFormat.DICT_LIST
    assert normalize_format("torch") == DataFormat.TORCH
    assert normalize_format(DataFormat.NUMPY) == DataFormat.NUMPY

def test_normalize_format_invalid():
    with pytest.raises(ValueError):
        normalize_format("invalid_format")


# ----------------------------------------------------
# Tests for to_python
# ----------------------------------------------------

def test_to_python_numpy_and_scalars():
    assert to_python(np.array([1, 2, 3])) == [1, 2, 3]
    assert isinstance(to_python(np.float32(3.14)), float)
    assert isinstance(to_python(np.int64(5)), int)

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_to_python_torch_tensor():
    t = torch.tensor([1, 2])
    assert to_python(t) == [1, 2]
    scalar_t = torch.tensor(42)
    assert to_python(scalar_t) == 42

@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_to_python_tf_tensor():
    t = tf.constant([1, 2])
    assert to_python(t) == [1, 2]
    scalar_t = tf.constant(42)
    assert to_python(scalar_t) == 42

def test_to_python_nested_structures():
    obj = {"a": np.array([1, 2]), "b": [np.float32(3), 4]}
    result = to_python(obj)
    assert result == {"a": [1, 2], "b": [3.0, 4]}


# ----------------------------------------------------
# Tests for to_list
# ----------------------------------------------------

def test_to_list_basic():
    assert to_list(np.array([1, 2])) == [1, 2]
    assert to_list(5) == [5]
    assert to_list([1, 2]) == [1, 2]
    assert to_list((1, 2)) == [1, 2]

def test_to_list_with_dict_raise():
    with pytest.raises(TypeError):
        to_list({"a": 1}, errors="raise")

def test_to_list_with_dict_coerce():
    result = to_list({"a": 1, "b": 2}, errors="coerce")
    assert result == [1, 2]

def test_to_list_with_dict_ignore():
    d = {"a": 1}
    assert to_list(d, errors="ignore") is d

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_to_list_torch():
    t = torch.tensor([1, 2, 3])
    assert to_list(t) == [1,2,3]
    
@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_to_list_tf():
    t = to_tensorflow([1, 2, 3])
    assert to_list(t) == [1,2,3]
    


# ----------------------------------------------------
# Tests for to_numpy
# ----------------------------------------------------

def test_to_numpy_valid():
    arr = to_numpy([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [1, 2, 3])

def test_to_numpy_scalar_and_coerce():
    assert np.allclose(to_numpy(5, errors="coerce"), np.array([5]))

def test_to_numpy_invalid_raises():
    class Custom:
        pass
    with pytest.raises(TypeError):
        to_numpy(Custom(), errors="raise")

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_to_numpy_torch():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert np.allclose(to_numpy(t), np.asarray([1,2,3]))
    
@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_to_numpy_tf():
    t = to_tensorflow([1, 2, 3])
    assert np.allclose(to_numpy(t), np.asarray([1,2,3]))


# ----------------------------------------------------
# Tests for to_torch
# ----------------------------------------------------

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_to_torch_valid():
    t = to_torch([1, 2, 3])
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_to_torch_scalar_and_coerce():
    t = to_torch(5, errors="coerce")
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([5.0]))


# ----------------------------------------------------
# Tests for to_tensorflow
# ----------------------------------------------------

@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_to_tensorflow_valid():
    t = to_tensorflow([1, 2, 3])
    assert isinstance(t, tf.Tensor)
    assert np.allclose(t.numpy(), [1.0, 2.0, 3.0])

@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_to_tensorflow_scalar_and_coerce():
    t = to_tensorflow(5, errors="coerce")
    assert np.allclose(t.numpy(), [5.0])



# ----------------------------------------------------
# Tests for convert_dict_to_format
# ----------------------------------------------------

@pytest.fixture
def sample_data():
    return {"a": np.array([1, 2, 3]), "b": [4, 5, 6]}

def test_convert_dict_to_format_dict(sample_data):
    result = convert_dict_to_format(sample_data, "dict")
    assert isinstance(result, dict)
    assert all(isinstance(v, list) for v in result.values())

def test_convert_dict_to_format_dict_list(sample_data):
    result = convert_dict_to_format(sample_data, "dict_list")
    assert isinstance(result, dict)
    assert all(isinstance(v, list) for v in result.values())

def test_convert_dict_to_format_dict_numpy(sample_data):
    result = convert_dict_to_format(sample_data, "dict_numpy")
    assert all(isinstance(v, np.ndarray) for v in result.values())

def test_convert_dict_to_format_pandas(sample_data):
    df = convert_dict_to_format(sample_data, "pandas")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]

def test_convert_dict_to_format_numpy(sample_data):
    arr = convert_dict_to_format(sample_data, "numpy")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)

def test_convert_dict_to_format_list(sample_data):
    result = convert_dict_to_format(sample_data, "list")
    assert isinstance(result, list)
    assert all(isinstance(row, list) for row in result)

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_convert_dict_to_format_torch(sample_data):
    t = convert_dict_to_format(sample_data, "torch")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 2)

@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_convert_dict_to_format_tensorflow(sample_data):
    t = convert_dict_to_format(sample_data, "tensorflow")
    assert isinstance(t, tf.Tensor)
    assert t.shape == (3, 2)


# ----------------------------------------------------
# Tests for get_data_format_for_backend
# ----------------------------------------------------

def test_get_data_format_for_backend():
    assert get_data_format_for_backend(Backend.TORCH) == DataFormat.TORCH
    assert get_data_format_for_backend(Backend.TENSORFLOW) == DataFormat.TENSORFLOW
    assert get_data_format_for_backend(Backend.SCIKIT) == DataFormat.NUMPY

    with pytest.raises(ValueError):
        get_data_format_for_backend("invalid_backend")
        
