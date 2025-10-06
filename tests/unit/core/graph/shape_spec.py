import pytest

from modularml.core.graph.shape_spec import ShapeSpec

# ==========================================================
# Basic property tests
# ==========================================================


def test_unique_shapes():
    spec = ShapeSpec({"a": (1, 32), "b": (1, 32), "c": (1, 16)})
    assert spec.unique_shapes == {(1, 32), (1, 16)}


def test_single_shape_properties():
    spec = ShapeSpec({"a": (10, 4)})
    assert spec.merged_axis == -1
    assert spec.merged_shape == (10, 4)


def test_empty_shape_spec_raises():
    spec = ShapeSpec({})
    with pytest.raises(ValueError, match="empty ShapeSpec"):
        _ = spec.merged_shape


# ==========================================================
# Merged axis inference
# ==========================================================


@pytest.mark.parametrize(
    ("shapes", "expected_axis"),
    [
        ({"a": (1, 32), "b": (1, 16)}, 1),
        ({"a": (4,), "b": (6,)}, 0),
        ({"a": (1, 32), "b": (1, 32)}, -1),  # identical
        ({"a": (1, 32), "b": (2, 16)}, None),  # ambiguous
    ],
)
def test_merged_axis_inference(shapes, expected_axis):
    spec = ShapeSpec(shapes)
    assert spec.merged_axis == expected_axis


# ==========================================================
# Merged shape computation
# ==========================================================


def test_merged_shape_simple():
    spec = ShapeSpec({"a": (1, 32), "b": (1, 16)})
    assert spec.merged_shape == (1, 48)


def test_merged_shape_different_rank_raises():
    spec = ShapeSpec({"a": (1, 32), "b": (16,)})
    with pytest.raises(ValueError, match="Inconsistent ranks"):
        _ = spec.merged_shape


def test_merged_shape_incompatible_dims_raises():
    spec = ShapeSpec({"a": (2, 32), "b": (1, 16)})
    with pytest.raises(ValueError, match="Cannot determine a unique merge axis"):
        _ = spec.merged_shape


# ==========================================================
# Indexing and equality
# ==========================================================


def test_get_and_index_access():
    spec = ShapeSpec({"x": (8, 16)})
    assert spec["x"] == (8, 16)
    assert spec.get("x") == (8, 16)
    with pytest.raises(KeyError):
        _ = spec["nonexistent"]


def test_equality_and_hash():
    a = ShapeSpec({"a": (1, 32)})
    b = ShapeSpec({"a": (1, 32)})
    c = ShapeSpec({"a": (2, 32)})
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


# ==========================================================
# Cross-Spec merge compatibility
# ==========================================================


def test_compatible_merge_with_true_and_false():
    a = ShapeSpec({"x": (1, 32)})
    b = ShapeSpec({"x": (1, 64)})
    c = ShapeSpec({"y": (1, 32)})

    assert a.compatible_merge_with(b)  # same keys, single differing axis
    assert not a.compatible_merge_with(c)  # different keys


def test_infer_merge_axis_with():
    a = ShapeSpec({"x": (1, 32)})
    b = ShapeSpec({"x": (1, 64)})
    c = ShapeSpec({"x": (2, 32)})

    assert a.infer_merge_axis_with(b) == 1
    assert a.infer_merge_axis_with(c) == 0
    assert a.infer_merge_axis_with(a) == -1


def test_infer_merge_axis_with_incompatible():
    a = ShapeSpec({"x": (1, 32)})
    b = ShapeSpec({"y": (2, 32)})
    assert a.infer_merge_axis_with(b) is None


# ==========================================================
# Merged shape across two ShapeSpecs
# ==========================================================


def test_merged_shape_with_success():
    a = ShapeSpec({"x": (1, 32)})
    b = ShapeSpec({"x": (1, 64)})
    assert a.merged_shape_with(b) == (1, 96)


def test_merged_shape_with_failure():
    a = ShapeSpec({"x": (1, 32)})
    b = ShapeSpec({"y": (1, 16)})
    with pytest.raises(ValueError, match="Cannot merge incompatible ShapeSpecs"):
        _ = a.merged_shape_with(b)


# ==========================================================
# Edge cases
# ==========================================================


def test_merged_axis_multiple_differences_is_none():
    spec = ShapeSpec({"a": (2, 32), "b": (1, 16)})
    assert spec.merged_axis is None


def test_repr_contains_keys():
    spec = ShapeSpec({"v": (1, 32)})
    rep = repr(spec)
    assert "v" in rep
    assert "ShapeSpec" in rep
