import numpy as np
import pytest

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.utils.modeling import (
    PadMode,
    make_dummy_batch,
    make_dummy_data,
    map_pad_mode_to_backend,
)


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


def test_map_pad_mode_to_backend_invalid_mode():
    with pytest.raises(ValueError, match=r"Pad mode .* not supported"):
        map_pad_mode_to_backend(PadMode.CIRCULAR, "tensorflow")


def test_map_pad_mode_to_backend_invalid_backend():
    with pytest.raises(ValueError, match="'invalid' is not a valid Backend"):
        map_pad_mode_to_backend(PadMode.CONSTANT, "invalid")


def test_make_dummy_data_shape_and_type():
    d = make_dummy_data((2, 3))
    assert isinstance(d, Data)
    arr = d.to_numpy() if hasattr(d, "to_numpy") else np.asarray(d.data)
    assert arr.shape == (2, 3)
    assert np.all(arr == 1.0)


def test_make_dummy_batch_structure():
    batch = make_dummy_batch(feature_shape=(2, 4), target_shape=(1, 2), batch_size=5)
    assert isinstance(batch, Batch)
    # Should contain one role "default"
    assert "default" in batch.role_samples
    sample_coll = batch.role_samples["default"]
    assert len(sample_coll) == 5  # batch size
    first = sample_coll[0]
    # Features and targets should exist
    assert all(key.startswith("features_") for key in first.features)
    assert all(key.startswith("targets_") for key in first.targets)
    assert "tags_1" in first.tags
    assert "tags_2" in first.tags
    # Shapes should match requested
    feat = next(iter(first.features.values()))
    targ = next(iter(first.targets.values()))
    assert isinstance(feat, Data)
    assert isinstance(targ, Data)


def test_make_dummy_batch_different_sizes():
    # Larger shapes
    batch = make_dummy_batch(feature_shape=(3, 5), target_shape=(2, 2), batch_size=2)
    sample_coll = batch.role_samples["default"]
    assert len(sample_coll) == 2
    first = sample_coll[0]
    assert len(first.features) == 3
    assert len(first.targets) == 2
