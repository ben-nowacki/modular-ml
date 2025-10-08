import numpy as np
import pytest

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.core.graph.shape_spec import ShapeSpec
from modularml.utils.modeling import (
    PadMode,
    make_dummy_batch,
    make_dummy_data,
    map_pad_mode_to_backend,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "backend", "expected"),
    [
        (PadMode.CONSTANT, "torch", "constant"),
        (PadMode.CONSTANT, "tensorflow", "constant"),
        (PadMode.REFLECT, "torch", "reflect"),
        (PadMode.REFLECT, "tensorflow", "reflect"),
        (PadMode.REFLECT, "scikit", "reflect"),
        (PadMode.REPLICATE, "torch", "replicate"),
        (PadMode.REPLICATE, "tensorflow", "SYMMETRIC"),
        (PadMode.REPLICATE, "scikit", "edge"),
        (PadMode.CIRCULAR, "torch", "circular"),
        (PadMode.CIRCULAR, "scikit", "wrap"),
    ],
)
def test_map_pad_mode_to_backend_valid(mode, backend, expected):
    assert map_pad_mode_to_backend(mode, backend) == expected


@pytest.mark.unit
def test_map_pad_mode_to_backend_invalid_mode():
    with pytest.raises(ValueError, match=r"Pad mode .* not supported"):
        map_pad_mode_to_backend(PadMode.CIRCULAR, "tensorflow")


@pytest.mark.unit
def test_map_pad_mode_to_backend_invalid_backend():
    with pytest.raises(ValueError, match="'invalid' is not a valid Backend"):
        map_pad_mode_to_backend(PadMode.CONSTANT, "invalid")


@pytest.mark.unit
def test_make_dummy_data_shape_and_type():
    d = make_dummy_data((2, 3))
    assert isinstance(d, Data)
    arr = d.to_numpy() if hasattr(d, "to_numpy") else np.asarray(d.data)
    assert arr.shape == (2, 3)
    assert np.all(arr == 1.0)


@pytest.mark.unit
def test_make_dummy_batch_structure():
    batch = make_dummy_batch(
        feature_shape=ShapeSpec({"X0": (2, 4)}),
        target_shape=ShapeSpec({"Y0": (1, 2)}),
        tag_shape=ShapeSpec({"TAG_0": (1,), "TAG_1": (1,)}),
        batch_size=5,
    )
    assert isinstance(batch, Batch)

    # Should contain one role "default"
    assert "default" in batch.role_samples
    sample_coll = batch.role_samples["default"]
    assert len(sample_coll) == 5  # batch size
    first = sample_coll[0]

    # Features and targets should exist
    assert all(key.startswith("X") for key in first.features)
    assert all(key.startswith("Y") for key in first.targets)
    assert "TAG_0" in first.tags
    assert "TAG_1" in first.tags

    # Shapes should match requested
    assert len(first.features) == 1
    assert len(first.targets) == 1
    assert len(first.tags) == 2

    # Ensure data type
    feat = next(iter(first.features.values()))
    targ = next(iter(first.targets.values()))
    assert isinstance(feat, Data)
    assert isinstance(targ, Data)


@pytest.mark.unit
def test_make_dummy_batch_different_sizes():
    # All features in a batch should have the same shape
    with pytest.raises(RuntimeError, match="Batch features have incompatible shapes"):
        batch = make_dummy_batch(
            feature_shape=ShapeSpec({"X0": (3, 5), "X1": (1, 100)}),
            target_shape=ShapeSpec({"Y0": (2, 2), "Y1": (2, 2)}),
            tag_shape=ShapeSpec({"TAG_0": (1,), "TAG_1": (1,)}),
            batch_size=5,
        )

    # Same with target in a batch should have the same shape
    with pytest.raises(RuntimeError, match="Batch targets have incompatible shapes"):
        batch = make_dummy_batch(
            feature_shape=ShapeSpec({"X0": (3, 5), "X1": (3, 5)}),
            target_shape=ShapeSpec({"Y0": (2, 2), "Y1": (1, 10)}),
            tag_shape=ShapeSpec({"TAG_0": (1,), "TAG_1": (1,)}),
            batch_size=5,
        )

    # All features in a batch should have the same shape
    batch = make_dummy_batch(
        feature_shape=ShapeSpec({"X0": (10,), "X1": (10,)}),
        target_shape=ShapeSpec({"Y0": (1,), "Y1": (1,)}),
        tag_shape=ShapeSpec({"TAG_0": (1,), "TAG_1": (1,)}),
        batch_size=5,
    )

    # Should contain one role "default"
    sample_coll = batch.role_samples["default"]
    assert len(sample_coll) == 5
    first = sample_coll[0]

    # Features and targets should exist
    assert all(key.startswith("X") for key in first.features)
    assert all(key.startswith("Y") for key in first.targets)
    assert "TAG_0" in first.tags
    assert "TAG_1" in first.tags

    # Shapes should match requested
    assert len(first.features) == 2
    assert len(first.targets) == 2
    assert len(first.tags) == 2

    # Ensure data type
    feat = next(iter(first.features.values()))
    targ = next(iter(first.targets.values()))
    assert isinstance(feat, Data)
    assert isinstance(targ, Data)
