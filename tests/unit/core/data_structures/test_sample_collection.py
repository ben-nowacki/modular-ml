import numpy as np
import pytest
import tensorflow as tf
import torch

from modularml.api import Backend, DataFormat
from modularml.core.api import Sample, SampleCollection
from modularml.core.graph.shape_spec import ShapeSpec
from tests.shared.data_utils import generate_dummy_sample


# ==========================================================
# Helper: make dummy samples
# ==========================================================
def make_sample_collection(n=3):
    """Generate a SampleCollection with `n` consistent samples."""
    samples = []
    for i in range(n):
        s = generate_dummy_sample(target_type="numeric")
        s.label = f"L{i:d}"
        samples.append(s)
    return SampleCollection(samples)


# ==========================================================
# Initialization
# ==========================================================
def test_init_valid():
    coll = make_sample_collection(5)
    assert isinstance(coll, SampleCollection)
    assert len(coll) == 5
    assert all(isinstance(s, Sample) for s in coll)


def test_init_invalid_type():
    with pytest.raises(TypeError, match="must be of type Sample"):
        SampleCollection(samples=[1, 2, 3])


def test_init_empty_collection():
    with pytest.raises(ValueError, match="at least one Sample"):
        SampleCollection(samples=[])


def test_inconsistent_feature_shape_raises():
    s1 = generate_dummy_sample(target_type="numeric", feature_shape_map={"X1": (1, 100)})
    s2 = generate_dummy_sample(target_type="numeric", feature_shape_map={"X1": (1, 50)})

    with pytest.raises(ValueError, match="Inconsistent SampleCollection feature shapes"):
        SampleCollection([s1, s2])


def test_inconsistent_target_shape_raises():
    s1 = generate_dummy_sample(target_type="numeric", target_shape_map={"T1": (1, 100)})
    s2 = generate_dummy_sample(target_type="numeric", target_shape_map={"T1": (1, 50)})

    with pytest.raises(ValueError, match="Inconsistent SampleCollection target shapes"):
        SampleCollection([s1, s2])


# ==========================================================
# Basic operations
# ==========================================================
def test_len_and_indexing():
    coll = make_sample_collection(5)
    assert len(coll) == 5
    s0 = coll[0]
    assert isinstance(s0, Sample)
    assert s0.label == "L0"


def test_get_by_uuid_and_label():
    coll = make_sample_collection(5)
    first_uuid = coll.samples[0].uuid
    first_label = coll.samples[0].label

    s_by_uuid = coll.get_sample_with_uuid(first_uuid)
    s_by_label = coll.get_sample_with_label(first_label)

    assert s_by_uuid.uuid == first_uuid
    assert s_by_label.label == first_label


def test_get_samples_with_uuid_subset():
    coll = make_sample_collection(5)
    uuids = [s.uuid for s in coll.samples[:2]]
    sub_coll = coll.get_samples_with_uuid(uuids)
    assert isinstance(sub_coll, SampleCollection)
    assert len(sub_coll) == 2


def test_get_samples_with_label_subset():
    coll = make_sample_collection(5)
    labels = [s.label for s in coll.samples[:2]]
    sub_coll = coll.get_samples_with_label(labels)
    assert isinstance(sub_coll, SampleCollection)
    assert len(sub_coll) == 2


def test_repr_and_iteration():
    coll = make_sample_collection(5)
    rep = repr(coll)
    assert "SampleCollection" in rep
    assert "n_samples=5" in rep
    items = list(iter(coll))
    assert len(items) == 5
    assert all(isinstance(s, Sample) for s in items)


# ==========================================================
# Property accessors
# ==========================================================
def test_property_accessors():
    coll = make_sample_collection(3)
    assert coll.feature_keys == ["X1", "X2"]
    assert coll.target_keys == ["Y1", "Y2"]
    assert coll.tag_keys == ["T_FLOAT", "T_STR"]
    assert len(coll.sample_uuids) == 3
    assert len(coll.sample_labels) == 3

    f_shape_spec = ShapeSpec(shapes={"X1": (1, 100), "X2": (1, 100)})
    assert coll.feature_shape_spec == f_shape_spec
    t_shape_spec = ShapeSpec(shapes={"Y1": (1, 1), "Y2": (1, 10)})
    assert coll.target_shape_spec == t_shape_spec


def test_get_feature_and_target_shape():
    coll = make_sample_collection(1)
    assert coll.get_feature_shape("X1") == (1, 100)
    assert coll.get_feature_shape("X2") == (1, 100)
    assert coll.get_target_shape("Y1") == (1, 1)
    assert coll.get_target_shape("Y2") == (1, 10)


# ==========================================================
# Data retrieval
# ==========================================================
def test_get_all_features_dict_numpy():
    coll = make_sample_collection(3)
    out = coll.get_all_features(fmt=DataFormat.DICT_NUMPY)
    assert "X1" in out
    assert isinstance(out["X1"], np.ndarray)
    assert out["X1"].shape[0] == len(coll)


def test_get_all_targets_and_tags():
    coll = make_sample_collection(3)
    out_targets = coll.get_all_targets(fmt=DataFormat.DICT_NUMPY)
    out_tags = coll.get_all_tags(fmt=DataFormat.DICT_NUMPY)
    assert "Y1" in out_targets
    assert "T_FLOAT" in out_tags
    assert isinstance(out_targets["Y1"], np.ndarray)
    assert len(out_tags["T_FLOAT"]) == len(coll)


def test_get_all_features_invalid_format_raises():
    coll = make_sample_collection(1)
    with pytest.raises(ValueError, match="Unknown data format: invalid"):
        coll.get_all_features(fmt="invalid")


# ==========================================================
# Backend conversion and copying
# ==========================================================
def test_to_backend_torch_and_tensorflow():
    coll = make_sample_collection(2)

    torch_coll = coll.to_backend(Backend.TORCH)
    tf_coll = coll.to_backend(Backend.TENSORFLOW)

    assert all(isinstance(s.features["X1"].value, torch.Tensor) for s in torch_coll)
    assert all(isinstance(s.features["X1"].value, tf.Tensor) for s in tf_coll)
    assert len(torch_coll) == len(coll)
    assert len(tf_coll) == len(coll)


def test_copy_creates_new_instances():
    coll = make_sample_collection(2)
    copied = coll.copy()
    assert isinstance(copied, SampleCollection)
    assert len(copied) == len(coll)
    # Ensure deep copy (different IDs)
    assert copied.samples[0] is not coll.samples[0]
    np.testing.assert_array_equal(copied.samples[0].features["X1"].value, coll.samples[0].features["X1"].value)
