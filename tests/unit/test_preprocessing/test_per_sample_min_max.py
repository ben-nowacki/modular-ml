import numpy as np
import pytest

from modularml.preprocessing.per_sample_min_max import PerSampleMinMaxScaler


@pytest.mark.unit
@pytest.mark.parametrize("feature_range", [(0, 1), (-1, 1), (2, 5)])
def test_per_sample_min_max_transform_and_inverse(feature_range):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 10))
    scaler = PerSampleMinMaxScaler(feature_range=feature_range).fit(X)
    Xt = scaler.transform(X)
    # Each sample should be within the feature_range
    mn, mx = feature_range
    assert np.all(Xt >= mn - 1e-9)
    assert np.all(Xt <= mx + 1e-9)
    # Inverse should reconstruct original (within tolerance)
    Xinv = scaler.inverse_transform(Xt)
    np.testing.assert_allclose(Xinv, X, atol=1e-8, rtol=1e-8)


@pytest.mark.unit
def test_per_sample_min_max_raises_on_bad_dim():
    scaler = PerSampleMinMaxScaler()
    with pytest.raises(ValueError, match=r"Expected 2D array"):
        scaler.transform(np.array([1.0, 2.0]))
