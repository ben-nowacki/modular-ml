import pytest

from modularml.core.graph.featureset import FeatureSet
from tests.shared.data_utils import generate_dummy_data, generate_dummy_featureset


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


# @pytest.fixture
# def battery_soh_featureset() -> FeatureSet:
#     """
#     Returns NMC 18650 aging dataset for SOH estimation.

#     Features = Pulses (101-element arrays)
#     Targets = SOH (scalar)
#     """
#     import pickle
#     import urllib.request
#     import warnings
#     from pathlib import Path

#     from modularml.core import FeatureTransform
#     from modularml.preprocessing import MinMaxScaler, PerSampleZeroStart

#     DATA_DIR = Path("tests/shared/data/battery_soh_featurset")
#     FS_PATH = DATA_DIR / "FeatureSet_NMC_charge_pulses"

#     if not FS_PATH.exists():
#         DATA_URL = "https://raw.githubusercontent.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation/main/processed_data/UConn-ILCC-NMC/data_slowpulse_1.pkl"
#         DATA_PATH = DATA_DIR / "NMC_raw_data.pkl"
#         DATA_DIR.mkdir(exist_ok=True, parents=True)

#         if not DATA_PATH.exists():
#             urllib.request.urlretrieve(url=DATA_URL, filename=DATA_PATH)  # noqa: S310

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=DeprecationWarning)
#             data = pickle.load(Path.open(DATA_PATH, "rb"))

#         fs = FeatureSet.from_dict(
#             label="PulseFeatures",
#             data={
#                 "voltage": data["voltage"],
#                 "soh": data["soh"],
#                 "cell_id": data["cell_id"],
#                 "group_id": data["group_id"],
#                 "pulse_type": data["pulse_type"],
#                 "pulse_soc": data["soc"],
#             },
#             feature_keys="voltage",
#             target_keys="soh",
#             tag_keys=["cell_id", "group_id", "pulse_type", "pulse_soc"],
#         )
#         fs_charge_only = fs.filter(pulse_type="chg")
#         fs_charge_only.label = "ChargePulses"

#         # Simple split
#         fs_charge_only.split_random(
#             ratios={"train": 0.5, "val": 0.3, "test": 0.2},
#             group_by="cell_id",
#         )

#         # Apply basic preprocessing (shift start of pulse to 0 and add MinMax)
#         fs_charge_only.fit_transform(
#             fit="train.features",
#             apply="features",
#             transform=FeatureTransform(PerSampleZeroStart()),
#         )
#         fs_charge_only.fit_transform(
#             fit="train.features",
#             apply="features",
#             transform=FeatureTransform(MinMaxScaler()),
#         )

#         # Normalize targets
#         fs_charge_only.fit_transform(
#             fit="train.targets",
#             apply="targets",
#             transform=FeatureTransform(MinMaxScaler()),
#         )
#         fs_charge_only.save(FS_PATH, overwrite_existing=True)

#     else:
#         fs_charge_only = FeatureSet.load(FS_PATH)

#     return fs_charge_only
