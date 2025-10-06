import pytest

from modularml.utils.backend import Backend, backend_requires_optimizer, infer_backend


@pytest.mark.unit
def test_backend_enum_values():
    assert Backend.TORCH == "torch"
    assert Backend.TENSORFLOW == "tensorflow"
    assert Backend.SCIKIT == "scikit"
    assert Backend.NONE == "none"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (Backend.TORCH, True),
        (Backend.TENSORFLOW, True),
        (Backend.SCIKIT, False),
        (Backend.NONE, False),
    ],
)
def test_backend_requires_optimizer(backend, expected):
    assert backend_requires_optimizer(backend) is expected


@pytest.mark.unit
def test_infer_backend_torch_tensor_and_model():
    torch = pytest.importorskip("torch")

    # Tensor
    t = torch.randn(2, 2)
    assert infer_backend(t) == Backend.TORCH

    # Module instance
    model = torch.nn.Linear(2, 2)
    assert infer_backend(model) == Backend.TORCH

    # Module class
    assert infer_backend(torch.nn.Linear) == Backend.TORCH


@pytest.mark.unit
def test_infer_backend_tensorflow_tensor_and_model():
    tf = pytest.importorskip("tensorflow")

    # Tensor
    t = tf.constant([[1.0, 2.0]])
    assert infer_backend(t) == Backend.TENSORFLOW

    # Keras model instance
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    assert infer_backend(model) == Backend.TENSORFLOW

    # Keras model class
    assert infer_backend(tf.keras.Sequential) == Backend.TENSORFLOW


@pytest.mark.unit
def test_infer_backend_scikit_estimator_and_class():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    # Estimator instance
    model = LinearRegression()
    assert infer_backend(model) == Backend.SCIKIT

    # Estimator class
    assert infer_backend(LinearRegression) == Backend.SCIKIT


@pytest.mark.unit
def test_infer_backend_none_for_unknown_objects():
    # Plain Python object
    assert infer_backend(object()) == Backend.NONE

    # Random class
    class Foo:
        pass

    assert infer_backend(Foo()) == Backend.NONE
    assert infer_backend(Foo) == Backend.NONE
