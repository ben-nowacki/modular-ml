import numpy as np
import pytest

from modularml.preprocessing.absolute import Absolute


@pytest.mark.unit
def test_absolute_transform_and_inverse_basic():
    X = np.array([[-1.0, 2.0, -3.5], [0.0, -4.2, 5.1]])
    t = Absolute().fit(X)
    Xt = t.transform(X)
    # transform should be absolute value
    np.testing.assert_allclose(Xt, np.abs(X))
    # inverse should restore original signs
    Xinv = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xinv, X)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_shape",
    [
        np.array([1.0, -2.0, 3.0]),  # 1D
        np.array([[[1.0, -2.0]]]),  # 3D
    ],
)
def test_absolute_raises_on_bad_dim(bad_shape):
    t = Absolute()
    with pytest.raises(ValueError, match=r"Expected 2D array"):
        t.transform(bad_shape)
    with pytest.raises(ValueError, match=r"Expected 2D array"):
        t.inverse_transform(bad_shape)
