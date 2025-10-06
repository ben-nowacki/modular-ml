import copy

import pytest

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.graph.shape_spec import ShapeSpec
from modularml.utils.backend import Backend


# ==========================================================
# Initialization and validation
# ==========================================================
def test_valid_sample_initialization(dummy_data_float):
    sample = Sample(
        features={"voltage": dummy_data_float},
        targets={"soh": dummy_data_float},
        tags={"cell": dummy_data_float},
        label="test_sample",
    )

    assert isinstance(sample.uuid, str)
    assert isinstance(sample.features["voltage"], Data)
    assert isinstance(sample.targets["soh"], Data)
    assert isinstance(sample.tags["cell"], Data)
    assert sample.label == "test_sample"


def test_invalid_uuid_type(dummy_data_float):
    with pytest.raises(TypeError, match="Sample ID must be a string"):
        Sample(
            features={"v": dummy_data_float},
            targets={"t": dummy_data_float},
            tags={"meta": dummy_data_float},
            uuid=123,  # invalid
        )


@pytest.mark.parametrize("field_name", ["features", "targets", "tags"])
def test_invalid_data_type_in_fields(dummy_data_float, field_name):
    kwargs = {
        "features": {"a": dummy_data_float},
        "targets": {"b": dummy_data_float},
        "tags": {"c": dummy_data_float},
    }
    kwargs[field_name] = {"invalid": 1234}  # not a Data instance
    with pytest.raises(TypeError, match="is not a Data object"):
        Sample(**kwargs)


# ==========================================================
# Property methods
# ==========================================================
def test_feature_target_tag_keys(dummy_sample_numeric):
    sample = dummy_sample_numeric
    assert isinstance(sample.feature_keys, list)
    assert all(isinstance(k, str) for k in sample.feature_keys)
    assert isinstance(sample.target_keys, list)
    assert isinstance(sample.tag_keys, list)


def test_shape_spec_properties(dummy_sample_numeric):
    fspec = dummy_sample_numeric.feature_shape_spec
    tspec = dummy_sample_numeric.target_shape_spec
    assert isinstance(fspec, ShapeSpec)
    assert isinstance(tspec, ShapeSpec)
    assert set(fspec.shapes.keys()) == set(dummy_sample_numeric.features.keys())
    assert set(tspec.shapes.keys()) == set(dummy_sample_numeric.targets.keys())


def test_get_feature_and_target_shapes(dummy_sample_numeric):
    key_f = next(iter(dummy_sample_numeric.features))
    key_t = next(iter(dummy_sample_numeric.targets))
    shape_f = dummy_sample_numeric.get_feature_shape(key_f)
    shape_t = dummy_sample_numeric.get_target_shape(key_t)
    assert isinstance(shape_f, tuple)
    assert isinstance(shape_t, tuple)


def test_getters_return_correct_data(dummy_sample_numeric):
    key_f = next(iter(dummy_sample_numeric.features))
    key_t = next(iter(dummy_sample_numeric.targets))
    key_tag = next(iter(dummy_sample_numeric.tags))
    assert isinstance(dummy_sample_numeric.get_features(key_f), Data)
    assert isinstance(dummy_sample_numeric.get_targets(key_t), Data)
    assert isinstance(dummy_sample_numeric.get_tags(key_tag), Data)


def test_repr_contains_label_and_id(dummy_sample_numeric):
    rep = repr(dummy_sample_numeric)
    assert "Sample(" in rep
    assert "features=" in rep
    assert "targets=" in rep
    assert "tags=" in rep
    assert dummy_sample_numeric.uuid[:8] in rep


# ==========================================================
# Backend conversion and copy
# ==========================================================
def test_to_backend_returns_new_sample(dummy_sample_numeric):
    sample = dummy_sample_numeric
    converted = sample.to_backend(Backend.TORCH)
    assert isinstance(converted, Sample)
    assert converted.uuid == sample.uuid
    assert all(isinstance(v, Data) for v in converted.features.values())


def test_copy_creates_independent_object(dummy_sample_numeric):
    sample = dummy_sample_numeric
    copied = sample.copy()
    assert copied is not sample
    assert copied.uuid == sample.uuid
    assert all(isinstance(v, Data) for v in copied.features.values())

    # Modify copied sample and ensure original unchanged
    some_key = next(iter(sample.features))
    sample.features[some_key].value = 100
    copied.features[some_key].value = 999
    assert sample.features[some_key].value == 100
    assert copied.features[some_key].value == 999


# ==========================================================
# Edge cases
# ==========================================================
def test_empty_fields_raise_type_error():
    """If any of features/targets/tags missing Data, raise immediately."""
    with pytest.raises(TypeError):
        Sample(features={}, targets={}, tags={"invalid": 123})
