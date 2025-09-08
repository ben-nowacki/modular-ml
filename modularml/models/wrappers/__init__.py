from typing import Any

from modularml.models.base import BaseModel
from modularml.utils.backend import Backend, infer_backend

from .scikit_wrapper import ScikitModelWrapper
from .tensorflow_wrapper import TensorflowModelWrapper
from .torch_wrapper import TorchModelWrapper


def wrap_model(model: Any) -> BaseModel:
    """
    Wraps a raw model object in a standardized BaseModel interface.

    This function allows users to pass native models from supported ML backends
    (e.g., PyTorch, TensorFlow, scikit-learn) and wraps them in a subclass of
    `BaseModel`, enabling consistent training, evaluation, and inference behavior
    across backends within the ModularML framework.

    If the input model is already an instance of `BaseModel`, it is returned as-is.
    Otherwise, the backend is inferred, and the model is wrapped using the appropriate
    wrapper class.

    Supported input types:
        - PyTorch: `torch.nn.Module` → wrapped with `TorchModelWrapper`
        - TensorFlow: `tf.keras.Model` → wrapped with `TensorflowModelWrapper`
        - scikit-learn: `BaseEstimator` → wrapped with `ScikitModelWrapper`

    Args:
        model (Any): The model to wrap. Can be a raw model instance from a supported backend
                     or already a `BaseModel`.

    Returns:
        BaseModel: A wrapped model instance conforming to the `BaseModel` interface.

    Raises:
        NotImplementedError: If the model's backend is unsupported or cannot be inferred.

    """
    # If already a BaseModel, can just return
    if isinstance(model, BaseModel):
        return model

    # Otherwise, try to infer backend and cast appropriately
    backend = infer_backend(model)

    if backend == Backend.TORCH:
        import torch  # noqa: PLC0415

        if issubclass(model, torch.nn.Module) or isinstance(model, torch.nn.Module):
            return TorchModelWrapper(model)
        msg = "Received PyTorch backend but model is not a subclass or instance of `torch.nn.Module`."
        raise ValueError(msg)

    if backend == Backend.TENSORFLOW:
        import tensorflow as tf  # noqa: PLC0415

        if issubclass(model, tf.keras.Model) or isinstance(model, tf.keras.Model):
            return TensorflowModelWrapper(model)
        msg = "Received Tensorflow backend but model is not a subclass or instance of `tf.keras.Model`."
        raise ValueError(msg)

    if backend == Backend.SCIKIT:
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        if issubclass(model, BaseEstimator) or isinstance(model, BaseEstimator):
            return ScikitModelWrapper(model)
        msg = "Received Scikit backend but model is not a subclass or instance of `sklearn.base.BaseEstimator`."
        raise ValueError(msg)

    msg = f"No wrapper available for inferred backend: {backend}"
    raise NotImplementedError(msg)
