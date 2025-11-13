import numpy as np
import pytest

from modularml.core.transforms.scaler import Scaler
from modularml.core.transforms.scaler_record import ScalerRecord


@pytest.mark.unit
def test_scaler_record_roundtrip():
    # --- create scaler and fit ---
    X = np.array([[1.0], [2.0], [3.0]])
    scaler = Scaler("StandardScaler")
    scaler.fit(X)

    # --- original record ---
    record = ScalerRecord(
        order=5,
        domain="features",
        keys=("voltage",),
        variant_in="raw",
        variant_out="transformed",
        fit_split="train",
        merged_axes=(0, 1),
        flatten_meta={"original_shape": (3, 1), "merge_axes": (0,)},
        scaler_object=scaler,
    )

    # --- serialize ---
    state = record.get_state()

    # --- deserialize ---
    restored = ScalerRecord.__new__(ScalerRecord)
    restored.set_state(state)

    # --- check basic fields ---
    assert restored.order == record.order
    assert restored.domain == "features"
    assert restored.keys == ("voltage",)
    assert restored.variant_in == "raw"
    assert restored.variant_out == "transformed"
    assert restored.fit_split == "train"
    assert restored.merged_axes == (0, 1)
    assert restored.flatten_meta == {"original_shape": (3, 1), "merge_axes": (0,)}

    # --- check scaler ---
    assert isinstance(restored.scaler_object, Scaler)
    assert restored.scaler_object._is_fit is True

    # check numerical correctness
    X_restored = restored.scaler_object.transform(X)
    X_expected = scaler.transform(X)
    assert np.allclose(X_restored, X_expected)
