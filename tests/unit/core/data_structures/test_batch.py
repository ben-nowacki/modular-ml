import numpy as np
import pytest

from modularml.core.api import Data
from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.sample_collection import SampleCollection
from tests.shared.data_utils import rng


# ==========================================================
# Batch construction and validation
# ==========================================================
@pytest.mark.unit
def test_batch_init_valid(dummy_featureset_numeric):
    """A Batch should initialize correctly with consistent shapes."""
    fs = dummy_featureset_numeric
    samples = fs.samples[:10]
    sc = SampleCollection(samples)
    batch = Batch(role_samples={"anchor": sc, "positive": sc})
    assert isinstance(batch, Batch)
    assert set(batch.available_roles) == {"anchor", "positive"}
    assert isinstance(batch.feature_shape, tuple)
    assert isinstance(batch.target_shape, tuple)
    assert len(batch) == len(sc)
    assert all(r in batch.role_sample_weights for r in batch.available_roles)


@pytest.mark.unit
def test_batch_inconsistent_feature_shape_raises(dummy_featureset_numeric):
    """Batch should raise ValueError if feature shapes differ across roles."""
    fs = dummy_featureset_numeric
    sc1 = SampleCollection(fs.samples[:5])
    sc2 = SampleCollection(fs.samples[5:10])

    # forcibly alter shape in one sample
    bad_sample = fs.samples[0].copy()
    key = bad_sample.feature_keys[0]
    bad_sample.features[key].value = rng.random(size=(5, 5))

    with pytest.raises(ValueError, match="Inconsistent SampleCollection feature shapes:"):
        _ = SampleCollection([bad_sample, *fs.samples[1:6]])

    sc1.samples[0].features[key].value = rng.random(size=(5, 5))
    with pytest.raises(RuntimeError, match="Cannot determine a unique merge axis for shapes"):
        _ = Batch(role_samples={"a": sc1, "b": sc2})


@pytest.mark.unit
def test_batch_inconsistent_target_shape_raises(dummy_featureset_numeric):
    """Batch should raise RuntimeError if target shapes differ across roles."""
    fs = dummy_featureset_numeric
    sc = SampleCollection(fs.samples[:5])

    # make a version with different target shape
    bad_sample = fs.samples[0].copy()
    key = bad_sample.target_keys[0]
    bad_sample.targets[key].value = rng.random(size=(5, 5))

    with pytest.raises(ValueError, match="Inconsistent SampleCollection target shapes:"):
        _ = SampleCollection([bad_sample, *fs.samples[1:6]])

    sc.samples[0].targets[key].value = rng.random(size=(5, 5))
    with pytest.raises(RuntimeError, match="Cannot determine a unique merge axis for shapes"):
        _ = Batch(role_samples={"a": sc})


@pytest.mark.unit
def test_batch_missing_weight_key_raises(dummy_featureset_numeric):
    """Batch should raise KeyError if weights missing for any role."""
    fs = dummy_featureset_numeric
    sc = SampleCollection(fs.samples[:5])
    weights = {"anchor": Data(np.ones(5))}  # missing 'pos'
    with pytest.raises(KeyError, match="missing required role"):
        Batch(role_samples={"anchor": sc, "pos": sc}, role_sample_weights=weights)


@pytest.mark.unit
def test_batch_weight_length_mismatch_raises(dummy_featureset_numeric):
    """Batch should raise ValueError if sample weights length mismatched."""
    fs = dummy_featureset_numeric
    sc = SampleCollection(fs.samples[:5])
    weights = {"anchor": Data(np.ones(4)), "pos": Data(np.ones(5))}
    with pytest.raises(ValueError, match="does not match length"):
        Batch(role_samples={"anchor": sc, "pos": sc}, role_sample_weights=weights)


@pytest.mark.unit
def test_batch_properties(dummy_featureset_numeric):
    """Check feature/target shapes and len are derived correctly."""
    fs = dummy_featureset_numeric
    sc = SampleCollection(fs.samples[:3])
    b = Batch(role_samples={"a": sc})
    assert isinstance(b.feature_shape, tuple)
    assert isinstance(b.target_shape, tuple)
    assert len(b) == 3
    assert "a" in b.available_roles
    assert isinstance(b.get_samples("a"), SampleCollection)


# ==========================================================
# BatchOutput validation
# ==========================================================
def make_batch_output():
    """Helper to create a valid BatchOutput for testing."""
    x = rng.random(size=(3, 2))
    y = rng.random(size=(3, 1))
    uuids = [f"uuid_{i}" for i in range(3)]
    return BatchOutput(
        features={"anchor": x, "positive": x},
        targets={"anchor": y, "positive": y},
        sample_uuids={"anchor": uuids, "positive": uuids},
    )


@pytest.mark.unit
def test_batchoutput_init_valid():
    bo = make_batch_output()
    assert isinstance(bo, BatchOutput)
    assert set(bo.available_roles) == {"anchor", "positive"}
    assert isinstance(bo.feature_shape, tuple)
    assert isinstance(bo.target_shape, tuple)


@pytest.mark.unit
def test_batchoutput_inconsistent_feature_shape_raises():
    # Ensure feature shapes are consistent across roles
    x1 = rng.random(size=(3, 2))
    x2 = rng.random(size=(4, 2))
    uuids = [f"u{i}" for i in range(3)]
    with pytest.raises(ValueError, match="Inconsistent feature shapes"):
        BatchOutput(features={"a": x1, "b": x2}, sample_uuids={"a": uuids, "b": uuids})


@pytest.mark.unit
def test_batchoutput_inconsistent_target_shape_raises():
    # Ensure targets shapes are consistent across roles
    x = rng.random(size=(3, 2))
    y1 = rng.random(size=(3, 1))
    y2 = rng.random(size=(4, 1))
    uuids = [f"u{i}" for i in range(3)]
    with pytest.raises(ValueError, match="Inconsistent target shapes"):
        BatchOutput(
            features={"a": x, "b": x},
            targets={"a": y1, "b": y2},
            sample_uuids={"a": uuids, "b": uuids},
        )


@pytest.mark.unit
def test_batchoutput_mismatched_keys_raises():
    x = rng.random(size=(3, 2))
    uuids = [f"u{i}" for i in range(3)]
    with pytest.raises(KeyError, match="Role names do not match"):
        BatchOutput(features={"a": x}, sample_uuids={"b": uuids})
