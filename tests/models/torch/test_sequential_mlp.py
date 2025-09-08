import pytest
import torch

from modularml.models.torch import SequentialMLP
from modularml.utils.backend import Backend


@pytest.fixture
def input_shape():
    return (16,)  # Flattened vector input


@pytest.fixture
def output_shape():
    return (8,)  # Output vector size


def test_eager_build_forward_pass(input_shape, output_shape):
    model = SequentialMLP(
        input_shape=input_shape,
        output_shape=output_shape,
        n_layers=3,
        hidden_dim=64,
        activation="relu",
        dropout=0.1,
    )
    assert model.is_built
    assert model.input_shape == input_shape
    assert model.output_shape == output_shape

    x = torch.randn(4, *input_shape)
    y = model(x)

    assert y.shape == (4, *output_shape)


def test_lazy_build_on_forward(input_shape, output_shape):
    model = SequentialMLP(
        n_layers=2,
        hidden_dim=32,
        activation="relu",
        dropout=0.0,
    )

    assert not model.is_built
    x = torch.randn(2, *input_shape)
    with pytest.warns(UserWarning, match="No output shape provided."):
        y = model(x)
    assert model.is_built
    assert model.input_shape == input_shape
    assert y.shape[1:] == model.output_shape


def test_shape_mismatch_raises(input_shape, output_shape):
    model = SequentialMLP(
        input_shape=input_shape,
        output_shape=(10,),  # Different from expected
        n_layers=2,
        hidden_dim=32,
    )
    assert model.is_built

    with pytest.raises(ValueError, match="Inconsistent output_shape"):
        model.build(input_shape=input_shape, output_shape=output_shape)


def test_output_shape_fallback(input_shape):
    model = SequentialMLP(
        input_shape=input_shape,
        n_layers=2,
        hidden_dim=32,
    )

    with pytest.warns(UserWarning, match="No output shape provided."):
        model.build()  # no output shape provided, should utilize hidden_dim
    assert model.output_shape == (1, 32)


def test_forward_shape_and_type(input_shape, output_shape):
    model = SequentialMLP(
        input_shape=input_shape,
        output_shape=output_shape,
        n_layers=3,
        hidden_dim=64,
        activation="gelu",
        dropout=0.2,
    )
    x = torch.randn(6, *input_shape)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (6, *output_shape)


def test_backend_is_torch():
    model = SequentialMLP()
    assert model.backend == Backend.TORCH
