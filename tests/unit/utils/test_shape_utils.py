import numpy as np
import pytest

from modularml.utils.optional_imports import ensure_pandas
from modularml.utils.shape_utils import (
    get_shape,
    shape_to_tuple,
    shapes_similar_except_singleton,
)


# ---------------------------------------------------------------------
# shapes_similar_except_singleton
# ---------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((32, 1, 4), (32, 4), True),
        ((1, 64, 1, 1), (64,), True),
        ((16, 8, 4), (8, 16, 4), False),
        ((32, 2, 4), (32, 4), False),
        ((1,), (1,), True),
        ((1,), (), True),
        ((3,), (), False),
    ],
)
def test_shapes_similar_except_singleton(a, b, expected):
    assert shapes_similar_except_singleton(a, b) == expected


# ---------------------------------------------------------------------
# shape_to_tuple
# ---------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2, 3], (1, 2, 3)),
        ((4, 5, 6), (4, 5, 6)),
        (np.array([7, 8, 9]), (7, 8, 9)),
        ([], ()),
    ],
)
def test_shape_to_tuple(value, expected):
    assert shape_to_tuple(value) == expected
    assert isinstance(shape_to_tuple(value), tuple)


# ---------------------------------------------------------------------
# get_shape — scalars and numpy
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_shape_scalars_and_numpy():
    assert get_shape(42) == ()
    assert get_shape("abc") == ()
    assert get_shape(None) == ()

    arr = np.zeros((3, 4, 5))
    assert get_shape(arr) == (3, 4, 5)


# ---------------------------------------------------------------------
# get_shape — sequences
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_shape_sequences():
    data = [[1, 2], [3, 4]]
    assert get_shape(data) == (2, 2)

    ragged = [[1, 2], [3]]
    assert get_shape(ragged) == (2,)

    nested_dicts = [{"a": 1}, {"a": 2}]
    shape = get_shape(nested_dicts)
    assert isinstance(shape, dict)
    assert "__shape__" in shape
    assert shape["__shape__"] == (2,)


@pytest.mark.unit
def test_get_shape_empty_sequence():
    assert get_shape([]) == (0,)


# ---------------------------------------------------------------------
# get_shape — objects with .shape
# ---------------------------------------------------------------------
class DummyWithShape:
    def __init__(self, shape):
        self.shape = shape


@pytest.mark.unit
def test_get_shape_shape_attribute_tuple():
    dummy = DummyWithShape((10, 20))
    assert get_shape(dummy) == (10, 20)


@pytest.mark.unit
def test_get_shape_shape_attribute_int():
    dummy = DummyWithShape(64)
    assert get_shape(dummy) == (64,)


@pytest.mark.unit
def test_get_shape_shape_attribute_callable():
    class CallableShape:
        def shape(self):
            return (5, 5)

    dummy = CallableShape()
    assert get_shape(dummy) == (5, 5)


# ---------------------------------------------------------------------
# get_shape — Pandas
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_shape_pandas():
    pd = ensure_pandas()
    s = pd.Series([[1, 2], [3, 4], [5, 6]])
    assert get_shape(s) == (3, 2)

    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [[1, 2, 3], [4, 5, 6]],
        },
    )
    shape = get_shape(df)
    assert shape["__shape__"] == (2, 2)
    assert "a" in shape["columns"]
    assert "b" in shape["columns"]
    assert shape["columns"]["b"] == (3,)


# ---------------------------------------------------------------------
# get_shape — TensorFlow and Torch (optional)
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_get_shape_torch_tensor():
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 3, 4)
    assert get_shape(x) == (2, 3, 4)


@pytest.mark.unit
def test_get_shape_tensorflow_tensor():
    tf = pytest.importorskip("tensorflow")
    x = tf.zeros((5, 6))
    assert get_shape(x) == (5, 6)
