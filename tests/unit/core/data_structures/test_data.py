import numpy as np
import pytest
import tensorflow as tf
import torch

from modularml.core.data_structures.data import Data
from modularml.utils.backend import Backend
from modularml.utils.data_format import infer_data_type


# ==========================================================
# Backend inference
# ==========================================================
def test_infer_backend_numpy():
    arr = np.array([1, 2, 3])
    d = Data(arr)
    assert d.backend == Backend.SCIKIT


def test_infer_backend_torch():
    t = torch.tensor([1.0, 2.0])
    d = Data(t)
    assert d.backend == Backend.TORCH


def test_infer_backend_tensorflow():
    t = tf.constant([1.0, 2.0])
    d = Data(t)
    assert d.backend == Backend.TENSORFLOW


def test_infer_backend_primitives():
    d = Data([1, 2, 3])
    assert d.backend == Backend.NONE

    d2 = Data(3.14)
    assert d2.backend == Backend.NONE

    d3 = Data(True)
    assert d3.backend == Backend.NONE


def test_infer_backend_invalid_type():
    class Dummy:
        pass

    with pytest.raises(TypeError, match="Unsupported type for Data"):
        Data(Dummy())


# ==========================================================
# Shape / dtype properties
# ==========================================================
def test_shape_and_dtype_numpy():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    d = Data(arr)
    assert d.shape == (2, 3)
    assert d.dtype == np.float64


def test_shape_and_dtype_torch():
    t = torch.ones((4, 2), dtype=torch.float32)
    d = Data(t)
    assert d.shape == (4, 2)
    assert d.dtype == torch.float32


def test_shape_and_dtype_tensorflow():
    t = tf.ones((3, 3), dtype=tf.float64)
    d = Data(t)
    assert d.shape == (3, 3)
    assert d.dtype == tf.float64


def test_shape_and_dtype_list():
    d = Data([[1, 2], [3, 4]])
    assert d.shape == (2, 2)
    assert d.dtype == list


# ==========================================================
# Indexing / length / equality
# ==========================================================
def test_len_and_getitem_numpy():
    arr = np.arange(10)
    d = Data(arr)
    assert len(d) == 10
    sliced = d[0:5]
    assert isinstance(sliced, Data)
    np.testing.assert_array_equal(sliced.value, arr[0:5])


def test_len_invalid_type():
    d = Data(5)
    with pytest.raises(TypeError, match="has no length"):
        _ = len(d)


def test_equality_and_comparison():
    d1 = Data(np.array([1, 2, 3]))
    d2 = Data(np.array([1, 2, 3]))
    assert d1 == d2
    assert d1 == d2


# ==========================================================
# Raw conversions
# ==========================================================
def test_to_numpy_from_torch():
    t = torch.tensor([[1.0, 2.0]])
    d = Data(t)
    np_arr = d.to_numpy()
    assert isinstance(np_arr, np.ndarray)
    np.testing.assert_allclose(np_arr, np.array([[1.0, 2.0]]))


def test_to_torch_from_numpy():
    arr = np.array([[1, 2]], dtype=np.float64)
    d = Data(arr)
    t = d.to_torch()
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([[1.0, 2.0]]))


def test_to_tensorflow_from_numpy():
    arr = np.array([[1.0, 2.0]])
    d = Data(arr)
    tf_t = d.to_tensorflow()
    assert isinstance(tf_t, tf.Tensor)
    np.testing.assert_allclose(tf_t.numpy(), arr)


# ==========================================================
# to_backend conversions (dtype inference)
# ==========================================================
@pytest.mark.parametrize(
    ("val", "expected_type"),
    [
        (np.array([1.0]), torch.float32),
        (np.array([1]), torch.int64),
        (np.array([True]), torch.bool),
    ],
)
def test_to_backend_torch_dtype_inference(val, expected_type):
    d = Data(val)
    out = d.to_backend(Backend.TORCH)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == expected_type


@pytest.mark.parametrize(
    ("val", "expected_type"),
    [
        (np.array([1.0]), tf.float32),
        (np.array([1]), tf.int64),
        (np.array([True]), tf.bool),
        (np.array(["a", "b"]), tf.string),
    ],
)
def test_to_backend_tensorflow_dtype_inference(val, expected_type):
    d = Data(val)
    out = d.to_backend(Backend.TENSORFLOW)
    assert isinstance(out, tf.Tensor)
    assert out.dtype == expected_type


def test_to_backend_scikit_dtype_inference():
    d = Data(np.array(["x", "y"]))
    out = d.to_backend(Backend.SCIKIT)
    assert isinstance(out, np.ndarray)
    assert out.dtype.type == np.str_


def test_to_backend_torch_string_raises():
    d = Data(np.array(["a", "b"]))
    with pytest.raises(TypeError, match="Cannot convert string data to PyTorch tensor"):
        d.to_backend(Backend.TORCH)


# ==========================================================
# Wrapped conversions
# ==========================================================
def test_as_numpy_and_torch_and_tf():
    arr = np.array([1.0, 2.0, 3.0])
    d = Data(arr)

    as_np = d.as_numpy()
    assert isinstance(as_np, Data)
    np.testing.assert_allclose(as_np.value, arr)

    as_torch = d.as_torch()
    assert isinstance(as_torch.value, torch.Tensor)
    assert torch.allclose(as_torch.value, torch.tensor(arr, dtype=torch.float32))

    as_tf = d.as_tensorflow()
    assert isinstance(as_tf.value, tf.Tensor)
    np.testing.assert_allclose(as_tf.value.numpy(), arr)


# ==========================================================
# Repr / hash / ordering
# ==========================================================
def test_repr_and_hash():
    d = Data(np.array([1.0]))
    rep = repr(d)
    assert "Data(backend=" in rep
    assert "dtype=" in rep
    _ = hash(d)  # should not raise


def test_comparison_operators():
    d1 = Data(5)
    d2 = Data(3)
    assert d1 > d2
    assert d2 < d1
    assert d1 >= d2
    assert d2 <= d1


# ==========================================================
# Integration: infer_data_type consistency
# ==========================================================
@pytest.mark.parametrize(
    ("val", "expected"),
    [
        (torch.tensor([1.0]), "float"),
        (torch.tensor([1]), "int"),
        (torch.tensor([True]), "bool"),
        (tf.constant([1.0]), "float"),
        (tf.constant(["a"], dtype=tf.string), "string"),
        (np.array([1.0]), "float"),
        (np.array(["a"]), "string"),
        (42, "int"),
        (3.14, "float"),
        (True, "bool"),
        ("a", "string"),
    ],
)
def test_infer_data_type_consistency(val, expected):
    inferred = infer_data_type(val)
    assert inferred == expected
