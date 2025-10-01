import numpy as np
import pytest
import tensorflow as tf
import torch

from modularml.core.loss import Loss
from modularml.utils.backend import Backend
from modularml.utils.exceptions import BackendNotSupportedError, LossError


# ------------------------
# Torch backend tests
# ------------------------
@pytest.mark.unit
def test_torch_loss_mse_forward_pass():
    loss = Loss(name="mse", backend=Backend.TORCH, reduction="mean")
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.5, 2.5])
    out = loss(x, y)
    assert torch.is_tensor(out)
    assert out.item() >= 0


@pytest.mark.unit
def test_torch_loss_invalid_name_raises():
    with pytest.raises(LossError, match="Unknown loss name"):
        Loss(name="not_a_real_loss", backend=Backend.TORCH)


# ------------------------
# TensorFlow backend tests
# ------------------------
@pytest.mark.unit
def test_tf_loss_mae_forward_pass():
    loss = Loss(name="mae", backend=Backend.TENSORFLOW, reduction="sum")
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.constant([1.0, 2.5, 2.5])
    out = loss(x, y)
    assert isinstance(out.numpy().item(), float)


@pytest.mark.unit
def test_tf_loss_invalid_name_raises():
    with pytest.raises(LossError, match="Unknown loss name"):
        Loss(name="foo", backend=Backend.TENSORFLOW)


# ------------------------
# Custom loss function
# ------------------------
def custom_numpy_loss(y_true, y_pred):
    return float(np.sum((y_true - y_pred) ** 2))


@pytest.mark.unit
def test_custom_loss_function_infers_backend_none():
    loss = Loss(loss_function=custom_numpy_loss)
    assert loss.backend == Backend.NONE
    out = loss(1, 1)
    assert out == 0.0


# ------------------------
# Error handling
# ------------------------
@pytest.mark.unit
def test_loss_requires_backend_or_function():
    with pytest.raises(LossError, match="Loss cannot be initiallized"):
        Loss()


@pytest.mark.unit
def test_backend_not_supported():
    with pytest.raises(BackendNotSupportedError, match=r"Loss._resolve"):
        Loss(name="mse", backend=Backend.SCIKIT)


@pytest.mark.unit
def test_call_wraps_exceptions():
    def bad_loss(*args, **kwargs):
        raise RuntimeError("boom")

    loss = Loss(loss_function=bad_loss)
    with pytest.raises(LossError, match="Failed to call loss function"):
        loss(1, 2)


# ------------------------
# Utility functions
# ------------------------
@pytest.mark.unit
def test_allowed_keywords_and_repr():
    loss = Loss(name="mse", backend=Backend.TORCH)
    assert "Loss(name='mse', backend=" in repr(loss)

    loss = Loss(loss_function=custom_numpy_loss)
    keys = loss.allowed_keywords
    assert any(k in keys for k in ["y_true", "y_pred"])
