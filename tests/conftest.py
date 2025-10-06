import pytest

from modularml.core.api import Data, FeatureSet, Sample
from tests.shared.data_utils import (
    generate_dummy_data,
    generate_dummy_featureset,
    generate_dummy_sample,
)


# ==========================================================
# Data primitives
# ==========================================================
@pytest.fixture
def dummy_data_float() -> Data:
    """A single scalar float Data object."""
    return Data(generate_dummy_data(shape=(), dtype="float"))


@pytest.fixture
def dummy_data_array() -> Data:
    """A 1D array Data object (length 10)."""
    return Data(generate_dummy_data(shape=(10,), dtype="float"))


@pytest.fixture
def dummy_data_str() -> Data:
    """A categorical string Data object."""
    return Data(generate_dummy_data(shape=(), dtype="str", choices=["red", "blue", "green"]))


# ==========================================================
# Sample-level
# ==========================================================
@pytest.fixture
def dummy_sample_numeric() -> Sample:
    """A Sample with numeric targets (for regression-style tests)."""
    return generate_dummy_sample(target_type="numeric")


@pytest.fixture
def dummy_sample_categorical() -> Sample:
    """A Sample with categorical targets (for classification-style tests)."""
    return generate_dummy_sample(target_type="categorical")


# ==========================================================
# FeatureSet-level
# ==========================================================
@pytest.fixture
def dummy_featureset_numeric() -> FeatureSet:
    """A FeatureSet with only numeric targets."""
    return generate_dummy_featureset(n_samples=1000, target_type="numeric", label="NumericFS")


@pytest.fixture
def dummy_featureset_categorical() -> FeatureSet:
    """A FeatureSet with only categorical targets."""
    return generate_dummy_featureset(n_samples=1000, target_type="categorical", label="CategoricalFS")
