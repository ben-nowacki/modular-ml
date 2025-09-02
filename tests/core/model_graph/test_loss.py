import uuid
import numpy as np
import torch
import pytest
from typing import Dict

from modularml.core.model_graph.loss import Loss, AppliedLoss, LossResult
from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.utils.backend import Backend


# --- Mock SampleCollection-like class ---
class MockSample:
    def __init__(self, features, targets):
        self._features = features
        self._targets = targets

    def get_all_features(self, format=None):
        return self._features

    def get_all_targets(self, format=None):
        return self._targets

    @property
    def feature_shape(self):
        return tuple(self._features.shape[1:]) if self._features is not None else ()

    @property
    def target_shape(self):
        return tuple(self._targets.shape[1:]) if self._targets is not None else ()

    def __len__(self):
        if self._features is not None:
            return self._features.shape[0]
        elif self._targets is not None:
            return self._targets.shape[0]
        else:
            return 0
        

# --- Test Function ---
def test_applied_loss_mse_torch():
    # Create dummy data
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    y_pred = torch.tensor([[1.1], [1.9], [2.9]])
    weights = np.ones(len(y_true))

    # Create Batch for FeatureSet input
    fs_batch = Batch(
        role_samples={'default': MockSample(features=None, targets=y_true)},
        role_sample_weights={'default': weights},
    )

    # Create dummy UUIDs
    sample_uuids = {'default': [str(uuid.uuid4()) for _ in range(len(y_true))]}

    # Create BatchOutput for ModelStage output
    model_output = BatchOutput(
        features={'default': y_pred},
        sample_uuids=sample_uuids,
    )

    # Define Loss and AppliedLoss
    loss = Loss(name='mse', backend=Backend.TORCH)
    applied = AppliedLoss(
        loss=loss,
        inputs={
            '0': 'PulseFeatures.targets',
            '1': 'Regressor.output'
        },
        label='mse_loss'
    )

    # Compute the loss
    result = applied.compute(
        batch_input={'PulseFeatures': fs_batch},
        model_outputs={'Regressor': model_output},
    )

    # Assertions
    assert isinstance(result, LossResult)
    assert result.label == 'mse_loss'
    assert isinstance(result.value, torch.Tensor)
    assert result.value.shape == torch.Size([])  # single scalar
    assert torch.isclose(result.value, torch.tensor(0.03, dtype=result.value.dtype), atol=1e-3)