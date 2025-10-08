from pathlib import Path

import pytest
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader, TensorDataset

import modularml as mml
from modularml.models.torch import SequentialMLP


@pytest.fixture(scope="session")
def model_config():
    """Shared model and optimizer config used across frameworks."""
    return {
        "input_shape": (101,),
        "output_shape": (1,),
        "hidden_dim": 32,
        "n_layers": 3,
        "lr": 1e-3,
        "batch_size": 32,
        "n_epochs": 10,
        "seed": 13,
    }


@pytest.fixture(scope="session")
def metrics():
    """Shared evaluation metrics."""

    def evaluate(y_true, y_pred):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    return evaluate


@pytest.fixture(scope="session")
def torch_model(model_config):
    """Creates a fresh SequentialMLP Torch model."""
    torch.manual_seed(model_config["seed"])
    model = SequentialMLP(
        input_shape=model_config["input_shape"],
        output_shape=model_config["output_shape"],
        hidden_dim=model_config["hidden_dim"],
        n_layers=model_config["n_layers"],
    )
    return model


# @pytest.fixture(scope="session")
# def dataset():
#     """Load ModularML FeatureSet and return pre-split Torch tensors."""
#     FILE_FEATURE_SET = Path("downloaded_data/charge_samples.joblib")
#     charge_samples = mml.core.FeatureSet.load(FILE_FEATURE_SET)

#     fss_train, fss_val, fss_test = (
#         charge_samples.get_subset("train"),
#         charge_samples.get_subset("val"),
#         charge_samples.get_subset("test"),
#     )

#     def to_tensors(fs):
#         X = torch.tensor(fs.get_all_features(fmt=mml.DataFormat.NUMPY), dtype=torch.float32)
#         y = torch.tensor(fs.get_all_targets(fmt=mml.DataFormat.NUMPY), dtype=torch.float32)
#         return X, y

#     return {
#         "train": to_tensors(fss_train),
#         "val": to_tensors(fss_val),
#         "test": to_tensors(fss_test),
#     }
