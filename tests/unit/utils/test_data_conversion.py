import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.data_conversion import (
    align_ranks,
    convert_to_format,
    enforce_numpy_shape,
    merge_dict_of_arrays_to_numpy,
    to_list,
    to_numpy,
    to_python,
    to_tensorflow,
    to_torch,
)
from modularml.utils.error_handling import ErrorMode

rng = np.random.default_rng(seed=13)


# ----------------------------------------------------------------------
# Fixtures and simple helpers
# ----------------------------------------------------------------------
@pytest.fixture
def base_data():
    """Base dictionary of simple (n_samples, n_features) arrays."""
    return {
        "a": np.arange(20).reshape(10, 2),
        "b": np.ones((10, 2)),
        "c": np.zeros((10, 2)),
    }


@pytest.fixture
def varied_rank_data_can_align():
    """Dictionary with arrays of different ranks for align_singletons testing."""
    # (10,4), (10,4,1,1), (10,4,1)
    return {
        "a": rng.random(size=(10, 4)),  # (10,4)
        "b": rng.random(size=(10, 4, 1, 1)),  # (10,4,1,1)
        "c": rng.random(size=(10, 4, 1)),  # (10,4,1)
    }


@pytest.fixture
def varied_rank_data_cannot_align():
    """Dictionary with arrays of different ranks for align_singletons testing."""
    return {
        "a": rng.random(size=(10, 4)),  # (10,4)
        "b": rng.random(size=(4,)),  # (10,)
        "c": rng.random(size=(10, 4, 1)),  # (10,4,1)
    }


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
    assert t.shape == (3,)
    tf_t = to_tensorflow(arr)
    assert isinstance(tf_t, tf.Tensor)
    assert tf_t.shape == (3,)


# ---------- merge_dict_of_arrays_to_numpy ----------
@pytest.mark.unit
def test_merge_dict_stack_basics(base_data):
    """Stack should add a new leading axis for dictionary keys."""
    out = merge_dict_of_arrays_to_numpy(base_data, mode="stack")
    assert out.shape == (3, 10, 2)
    assert np.allclose(out[1], np.ones((10, 2)))  # second element all ones


@pytest.mark.unit
def test_merge_dict_stack_with_custom_axis(base_data):
    """Custom axis stacking should insert new dimension at the correct position."""
    out = merge_dict_of_arrays_to_numpy(base_data, mode="stack", axis=1)
    # Shape: (10, 3, 2) — new axis inserted after sample dimension
    assert out.shape == (10, 3, 2)


@pytest.mark.unit
def test_merge_dict_stack_fails_on_mismatched_shapes():
    """Stack should raise if array shapes differ."""
    data = {"a": np.ones((10, 2)), "b": np.ones((5, 2))}
    with pytest.raises(ValueError, match="all input arrays must have the same shape"):
        merge_dict_of_arrays_to_numpy(data, mode="stack")


@pytest.mark.unit
def test_merge_dict_concat_mode_default_axis(base_data):
    """Concatenation along axis=0 should join samples."""
    out = merge_dict_of_arrays_to_numpy(base_data, mode="concat")
    assert out.shape == (30, 2)  # 10 samples x 3 arrays
    # Verify ordering: 'a' then 'b' then 'c'
    np.testing.assert_array_equal(out[:10], np.arange(20).reshape(10, 2))


@pytest.mark.unit
def test_merge_dict_concat_mode_feature_axis(base_data):
    """Concatenation along axis=-1 should join feature dimension."""
    out = merge_dict_of_arrays_to_numpy(base_data, mode="concat", axis=-1)
    assert out.shape == (10, 6)  # 3 x 2 features combined


@pytest.mark.unit
def test_merge_dict_concat_mode_incompatible_shapes():
    """Concat should raise for incompatible shapes."""
    data = {"a": np.ones((10, 2)), "b": np.ones((5, 2))}
    x = merge_dict_of_arrays_to_numpy(data, mode="concat", axis=0)
    assert x.shape == (15, 2)

    with pytest.raises(
        ValueError,
        match="all the input array dimensions except for the concatenation axis must match exactly",
    ):
        merge_dict_of_arrays_to_numpy(data, mode="concat", axis=1)


@pytest.mark.unit
def test_merge_dict_flatten_mode(base_data):
    """Flatten should concatenate feature dimensions horizontally."""
    # Flatten to 1D (3 * 10 * 2 = 60)
    out = merge_dict_of_arrays_to_numpy(base_data, mode="flatten", axis=None)
    assert out.shape == (60,)

    # Flatten after axis 0 -> 3 of (10, 2) -> (10, 6)
    out = merge_dict_of_arrays_to_numpy(base_data, mode="flatten", axis=0)
    assert out.shape == (10, 6)


@pytest.mark.unit
def test_merge_dict_flatten_with_different_ranks(
    varied_rank_data_can_align,
    varied_rank_data_cannot_align,
):
    # Align singletons, then flatten after axis 0
    # (10,4), (10,), (10,4,1) --> aligned to (10,4,1,1) --> (10, 4*3)
    out = merge_dict_of_arrays_to_numpy(
        varied_rank_data_can_align,
        mode="flatten",
        axis=0,
        align_singletons=True,
    )
    assert out.ndim == 2
    assert out.shape == (10, 12)

    # (10,4), (4,), (10,4,1) --> cannot align but flattening all so should pass
    out = merge_dict_of_arrays_to_numpy(varied_rank_data_cannot_align, mode="flatten", axis=None)
    assert out.shape == (84,)  # 10*4 + 4 + 10*4*1

    # Specifying any axis should fail since shapes don't align
    with pytest.raises(
        ValueError,
        match="all the input array dimensions except for the concatenation axis must match exactly",
    ):
        out = merge_dict_of_arrays_to_numpy(
            varied_rank_data_cannot_align,
            mode="flatten",
            axis=0,
            align_singletons=False,
        )


@pytest.mark.unit
def test_merge_dict_auto(base_data):
    # Should prefer stack, if possible
    out = merge_dict_of_arrays_to_numpy(base_data, mode="auto")
    assert out.shape == (3, 10, 2)

    # If not, fallback to concat on axis 0
    data = {"a": np.ones((5, 2)), "b": np.ones((4, 2))}
    out = merge_dict_of_arrays_to_numpy(data, mode="auto")
    assert out.shape == (9, 2)

    # If not, 2nd fallback to concat on axis -1
    data = {"a": np.ones((10, 2)), "b": np.ones((10, 3))}
    out = merge_dict_of_arrays_to_numpy(data, mode="auto")
    assert out.shape == (10, 5)

    # Else, flatten to 1D array
    data = {"a": np.ones((3, 2)), "b": np.ones((4, 5))}
    out = merge_dict_of_arrays_to_numpy(data, mode="auto")
    assert out.shape == (26,)  # 3 * 2 + 4 * 5

    # Fails on empyty input
    with pytest.raises(ValueError, match="No arrays provided"):
        merge_dict_of_arrays_to_numpy({})


@pytest.mark.unit
def test_merge_dict_invalid_mode_raises(base_data):
    """Invalid mode should raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        merge_dict_of_arrays_to_numpy(base_data, mode="badmode")


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
