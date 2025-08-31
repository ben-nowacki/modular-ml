import numpy as np
import torch
import pytest
from typing import Dict

from modularml.core.model_graph.loss import Loss, AppliedLoss, LossResult
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat

# --- Mocks ---
class MockSample:
    def __init__(self, features, targets):
        self._features = features
        self._targets = targets

    def get_all_features(self, format=DataFormat.NUMPY):
        return self._features

    def get_all_targets(self, format=DataFormat.NUMPY):
        return self._targets

class MockBatch:
    def __init__(self, role_samples: Dict[str, MockSample], role_sample_weights: Dict[str, np.ndarray]):
        self.role_samples = role_samples
        self.role_sample_weights = role_sample_weights

    @property
    def available_roles(self):
        return list(self.role_samples.keys())

# --- Test Function ---
def test_applied_loss_mse_torch():
    # Define sample data
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    y_pred = torch.tensor([[1.1], [1.9], [2.9]])
    
    # Create mock batches
    fs_batch = MockBatch(
        role_samples={'default': MockSample(features=None, targets=y_true)},
        role_sample_weights={'default': np.ones(len(y_true))}
    )

    model_output = MockBatch(
        role_samples={'default': MockSample(features=y_pred, targets=None)},
        role_sample_weights={'default': np.ones(len(y_pred))}
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
    
    # Compute loss
    result = applied.compute(
        batches={'PulseFeatures': fs_batch},
        model_outputs={'Regressor': model_output}
    )

    # Assertions
    assert isinstance(result, LossResult)
    assert result.label == 'mse_loss'
    assert isinstance(result.value, torch.Tensor)
    assert result.value.shape == torch.Size([3, 1]) 
    assert np.allclose(result.sample_weights, np.ones(3))
    assert result.weight == 1.0