import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from modularml.preprocessing.segmented_scaler import SegmentedScaler


@pytest.mark.unit
def test_segmented_scaler_transform_inverse():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 12))
    boundaries = (0, 5, 8, 12)
    seg = SegmentedScaler(boundaries=boundaries, scaler=MinMaxScaler()).fit(X)
    Xt = seg.transform(X)
    # Each segment individually between 0 and 1
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg_block = Xt[:, s:e]
        assert np.all(seg_block >= -1e-9)
        assert np.all(seg_block <= 1 + 1e-9)
    # Inverse recovers original
    Xinv = seg.inverse_transform(Xt)
    np.testing.assert_allclose(Xinv, X, atol=1e-8, rtol=1e-8)


@pytest.mark.unit
def test_segmented_scaler_validates_boundaries_and_shapes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 7))
    # Non-ascending boundaries
    with pytest.raises(ValueError, match=r"Boundaries must be strictly ascending"):
        SegmentedScaler(boundaries=(0, 5, 5, 7), scaler="standardscaler").fit(X)
    # Out of range endpoints
    with pytest.raises(ValueError, match=r"Boundaries must start at 0"):
        SegmentedScaler(boundaries=(1, 5, 7), scaler="standardscaler").fit(X)
    with pytest.raises(ValueError, match=r"Last boundary does not match feature length"):
        SegmentedScaler(boundaries=(0, 5, 6), scaler="standardscaler").fit(X)
