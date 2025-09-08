import pytest
import torch
from torch import nn

from modularml.models.wrappers import TorchModelWrapper


# === Dummy model classes for testing ===
class SimpleMLP(nn.Module):
    def __init__(self, custom_input_shape_key=(10,), custom_output_shape_key=(5,)):
        super().__init__()
        input_dim = int(torch.prod(torch.tensor(custom_input_shape_key)))
        output_dim = int(torch.prod(torch.tensor(custom_output_shape_key)))
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


# === Fixtures and shared parameters ===
@pytest.fixture
def input_shape():
    return (10,)  # 1D input vector of length 10


@pytest.fixture
def output_shape():
    return (5,)  # Output vector of length 5


# === Eager mode ===
def test_eager_instantiation_with_valid_shapes(input_shape, output_shape):
    """Test build and forward pass with a pre-instantiated model and correct shapes."""
    model = SimpleMLP(input_shape, output_shape)
    wrapper = TorchModelWrapper(model=model)
    wrapper.build(input_shape=input_shape, output_shape=output_shape)

    assert wrapper.input_shape == input_shape
    assert wrapper.output_shape == output_shape

    x = torch.randn(2, *input_shape)
    y = wrapper(x)
    assert y.shape == (2, *output_shape)


def test_eager_instantiation_with_output_shape_mismatch_raises(input_shape, output_shape):
    """Test that a shape mismatch raises an error in eager mode."""
    model = SimpleMLP(input_shape, (3,))
    wrapper = TorchModelWrapper(model=model)

    with pytest.raises(RuntimeError, match="does not match expected output_shape"):
        wrapper.build(input_shape=input_shape, output_shape=output_shape)


# === Lazy mode ===
def test_lazy_instantiation_with_shape_injection(input_shape, output_shape):
    """Test lazy instantiation with correct shape injection and valid forward pass."""
    wrapper = TorchModelWrapper(
        model_class=SimpleMLP,
        model_kwargs={},
        inject_input_shape_as="custom_input_shape_key",
        inject_output_shape_as="custom_output_shape_key",
    )
    wrapper.build(input_shape=input_shape, output_shape=output_shape)

    assert isinstance(wrapper.model, SimpleMLP)
    assert wrapper.input_shape == input_shape
    assert wrapper.output_shape == output_shape

    x = torch.randn(4, *input_shape)
    y = wrapper(x)
    assert y.shape == (4, *output_shape)


def test_lazy_instantiation_with_explicit_kwargs_override(input_shape, output_shape):
    """Test that explicitly passed kwargs override injection, and that shape mismatch causes a runtime error."""
    wrapper = TorchModelWrapper(
        model_class=SimpleMLP,
        model_kwargs={
            "custom_input_shape_key": (123,),
            "custom_output_shape_key": (321,),
        },
    )

    with (
        pytest.raises(RuntimeError, match="failed to accept the expected input_shape:"),
        pytest.warns(Warning),  # noqa: PT030
    ):
        wrapper.build(input_shape=input_shape, output_shape=output_shape)

    assert wrapper.model.linear.in_features == 123
    assert wrapper.model.linear.out_features == 321


def test_lazy_instantiation_with_incorrect_injection_keys(input_shape, output_shape):
    """Test that incorrect inject keys raise a ValueError when model_class does not accept them."""
    wrapper = TorchModelWrapper(
        model_class=SimpleMLP,
        inject_input_shape_as="custom_input_shape",
        inject_output_shape_as="custom_output_shape",
    )

    with pytest.raises(ValueError, match="not accepted by `SimpleMLP` constructor."):
        wrapper.build(input_shape=input_shape, output_shape=output_shape)


def test_lazy_instantiation_with_unused_shape_injection_warnings(input_shape, output_shape):
    """Test that unused shape injections raise RuntimeWarnings but do not crash."""
    wrapper = TorchModelWrapper(
        model_class=SimpleMLP,
        model_kwargs={
            "custom_input_shape_key": input_shape,
            "custom_output_shape_key": output_shape,
        },
    )

    with pytest.warns(Warning) as record:  # noqa: PT030
        wrapper.build(input_shape=input_shape, output_shape=output_shape)

    messages = [str(w.message) for w in record]
    assert any("does not accept `input_shape` as a kwarg" in m for m in messages)
    assert any("does not accept `output_shape` as a kwarg" in m for m in messages)
    assert all(issubclass(w.category, RuntimeWarning) for w in record)


def test_lazy_instantiation_with_correct_keys_no_warnings(input_shape, output_shape):
    """Test lazy instantiation with correctly specified shape injection keys and no warnings."""
    wrapper = TorchModelWrapper(
        model_class=SimpleMLP,
        inject_input_shape_as="custom_input_shape_key",
        inject_output_shape_as="custom_output_shape_key",
    )
    wrapper.build(input_shape=input_shape, output_shape=output_shape)


# === Input Validation Tests ===
def test_missing_model_and_class_raises():
    """Test error when neither model nor model_class is provided."""
    with pytest.raises(ValueError, match="Must provide either"):
        TorchModelWrapper()


def test_invalid_model_type_raises():
    """Test error when an invalid model instance is passed."""
    with pytest.raises(ValueError, match="must be an instance of torch.nn.Module"):
        TorchModelWrapper(model="not a model")


def test_invalid_model_class_type_raises():
    """Test error when model_class is not callable."""
    with pytest.raises(ValueError, match="model_class must be callable"):
        TorchModelWrapper(model_class="not callable")
