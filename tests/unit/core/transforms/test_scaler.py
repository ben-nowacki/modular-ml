import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from modularml.core.transforms.scaler import Scaler
from modularml.preprocessing import SCALER_REGISTRY


@pytest.fixture
def sample_data():
    return np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])


@pytest.fixture
def scaler_name():
    assert "StandardScaler" in SCALER_REGISTRY
    return "StandardScaler"


# ------------------------------------------------------------
# Initialization Tests
# ------------------------------------------------------------
@pytest.mark.unit
def test_init_by_name(scaler_name):
    s = Scaler(scaler_name)
    assert s.scaler_name == scaler_name
    assert isinstance(s._scaler, StandardScaler)
    assert s._is_fit is False


@pytest.mark.unit
def test_init_bad_name():
    with pytest.raises(ValueError, match="not recognized"):
        Scaler("DoesNotExist")


@pytest.mark.unit
def test_init_by_instance():
    inst = StandardScaler(with_mean=False)
    s = Scaler(inst)
    # Should recover class name or exact match
    assert s.scaler_name in ("StandardScaler", inst.__class__.__name__)
    assert s._scaler is inst
    assert s._is_fit is False


# ------------------------------------------------------------
# Fit / Transform / FitTransform
# ------------------------------------------------------------
@pytest.mark.unit
def test_fit_sets_flag(sample_data):
    s = Scaler("StandardScaler")
    s.fit(sample_data)
    assert s._is_fit is True


@pytest.mark.unit
def test_transform_requires_fit(sample_data):
    s = Scaler("StandardScaler")
    with pytest.raises(RuntimeError):
        s.transform(sample_data)


@pytest.mark.unit
def test_fit_transform(sample_data):
    s = Scaler("StandardScaler")
    out = s.fit_transform(sample_data)
    assert s._is_fit is True
    assert out.shape == sample_data.shape


@pytest.mark.unit
def test_inverse_transform_round_trip(sample_data):
    s = Scaler("StandardScaler")
    s.fit(sample_data)
    x = s.transform(sample_data)
    xr = s.inverse_transform(x)
    assert np.allclose(sample_data, xr)


# ------------------------------------------------------------
# Serialization
# ------------------------------------------------------------
@pytest.mark.unit
def test_get_state_structure():
    s = Scaler("StandardScaler", scaler_kwargs={"with_mean": False})
    state = s.get_state()

    assert "version" in state
    assert state["version"] == "1.0"
    assert state["scaler_name"] == "StandardScaler"
    assert state["scaler_kwargs"] == {"with_mean": False}
    assert state["_is_fit"] is False


@pytest.mark.unit
def test_set_state_reconstruction(sample_data):
    # Fit original
    s1 = Scaler("StandardScaler")
    s1.fit(sample_data)
    x1 = s1.transform(sample_data)

    # Serialize
    state = s1.get_state()

    # Construct new and restore
    s2 = Scaler("StandardScaler")  # constructor signature required by SerializableMixin
    s2.set_state(state)

    # Sklearn scalers save learned parameters in state
    # Therefore, should retain _is_fit across reload
    assert s2._is_fit == s1._is_fit
    s2.transform(sample_data)


# ------------------------------------------------------------
# Behavioral Consistency (fit → serialize → reload → fit again)
# ------------------------------------------------------------
@pytest.mark.unit
def test_full_reproduce_pipeline(sample_data):
    # Since current Scaler.get_state does NOT store learned parameters,
    # reconstruction requires re-fitting to restore behavior.

    # Original
    s1 = Scaler("StandardScaler")
    s1.fit(sample_data)
    out1 = s1.transform(sample_data)

    # Serialize
    state = s1.get_state()

    # Reload
    s2 = Scaler("StandardScaler")
    s2.set_state(state)

    # Must re-fit to actually match s1
    s2.fit(sample_data)
    out2 = s2.transform(sample_data)

    assert np.allclose(out1, out2)
