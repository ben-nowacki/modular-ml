import numpy as np
import pytest

from modularml.preprocessing.per_sample_zero import PerSampleZeroStart


@pytest.mark.unit
def test_per_sample_zero_transform_and_inverse():
    X = np.array([[5.0, 6.0, 7.5], [-3.0, -2.5, -1.0], [0.0, 1.0, 2.0]])
    t = PerSampleZeroStart().fit(X)
    Xt = t.transform(X)
    # first column should be zeros
    assert np.allclose(Xt[:, 0], 0.0)
    # differences preserved
    np.testing.assert_allclose(Xt[:, 1:] - Xt[:, [0]], X[:, 1:] - X[:, [0]])
    # inverse returns original
    Xinv = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xinv, X)


@pytest.mark.unit
def test_per_sample_zero_raises_on_bad_dim():
    t = PerSampleZeroStart()
    bad = np.array([1.0, 2.0, 3.0])  # 1D
    with pytest.raises(ValueError, match=r"Expected 2D array"):
        t.transform(bad)
    with pytest.raises(ValueError, match=r"Expected 2D array"):
        t.inverse_transform(bad)
