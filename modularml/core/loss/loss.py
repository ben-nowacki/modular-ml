import inspect
from collections.abc import Callable

import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.exceptions import BackendNotSupportedError, LossError


class Loss:
    """
    A backend-agnostic wrapper around loss functions used in model training.

    This class allows the use of built-in loss functions from supported backends (PyTorch, TensorFlow)
    or custom-defined loss functions (e.g., scikit-learn, numpy-based) and ensures compatibility with
    the modular training workflow.
    """

    def __init__(
        self,
        name: str | None = None,
        backend: Backend | None = None,
        loss_function: Callable | None = None,
        reduction: str = "none",
    ):
        """
        A backend-agnostic wrapper around loss functions used in model training.

        This class allows the use of built-in loss functions from supported backends (PyTorch, TensorFlow)
        or custom-defined loss functions (e.g., scikit-learn, numpy-based) and ensures compatibility with
        the modular training workflow.

        Args:
            name (str | None): Name of the built-in loss function (e.g., "mse", "mae").
            backend (Backend | None): Backend to use (e.g., Backend.TORCH or Backend.TENSORFLOW).
            loss_function (Callable | None): A custom user-defined loss function.
            reduction (str): Reduction strategy (e.g., "mean", "sum", "none"). Defaults to "none".

        Raises:
            LossError: If neither `loss_function` nor both `name` and `backend` are provided.
            BackendNotSupportedError: If backend resolution is attempted on an unsupported backend.

        Examples:
            ```python
            # Using built-in loss
            loss1 = Loss(name="mse", backend=Backend.TORCH)

            # Using custom loss function
            from sklearn.metrics import mean_squared_error

            loss2 = Loss(loss_function=mean_squared_error)
            ```

        """
        self.name = name.lower() if name else None
        self.backend = backend
        self.reduction = reduction

        if loss_function is not None:
            self.loss_function: Callable = loss_function
            mod = inspect.getmodule(loss_function)
            if self.name is None:
                self.name = mod.__name__

            # TODO: how to infer backend?
            if "torch" in mod.__name__:
                self.backend = Backend.TORCH
            elif "tensorflow" in mod.__name__:
                self.backend = Backend.TENSORFLOW
            else:
                self.backend = Backend.NONE

        elif name and backend:
            self.loss_function: Callable = self._resolve()
        else:
            msg = "Loss cannot be initiallized. You must specify either `loss_function` or both `name` and `backend`."
            raise LossError(msg)

    def _resolve(self) -> Callable:
        """
        Resolves the appropriate loss function object from the selected backend using the given name.

        Returns:
            Callable: A callable loss function.

        Raises:
            BackendNotSupportedError: If the backend is not supported.
            LossError: If the loss name is not recognized.

        """
        avail_losses = {}
        if self.backend == Backend.TORCH:
            avail_losses = {
                "mse": torch.nn.MSELoss(reduction=self.reduction),
                "mae": torch.nn.L1Loss(reduction=self.reduction),
                "cross_entropy": torch.nn.CrossEntropyLoss(reduction=self.reduction),
                "bce": torch.nn.BCELoss(reduction=self.reduction),
                "bce_logits": torch.nn.BCEWithLogitsLoss(reduction=self.reduction),
                "cosine_embedding": torch.nn.CosineEmbeddingLoss(reduction=self.reduction)
            }
        elif self.backend == Backend.TENSORFLOW:
            avail_losses = {
                "mse": tf.keras.losses.MeanSquaredError(reduction=self.reduction),
                "mae": tf.keras.losses.MeanAbsoluteError(reduction=self.reduction),
                "cross_entropy": tf.keras.losses.CategoricalCrossentropy(reduction=self.reduction),
                "bce": tf.keras.losses.BinaryCrossentropy(reduction=self.reduction),
            }
        else:
            raise BackendNotSupportedError(backend=self.backend, method="Loss._resolve()")

        loss = avail_losses.get(self.name)
        if loss is None:
            msg = (
                f"Unknown loss name (`{self.name}`) for `{self.backend}` backend."
                f"Available losses: {avail_losses.keys()}"
            )
            raise LossError(msg)
        return loss

    @property
    def allowed_keywords(self) -> list[str]:
        """
        Returns the list of valid keyword arguments for the current loss function.

        Returns:
            List[str]: A list of argument names accepted by the loss function.

        """
        # Get the signature object
        sig = inspect.signature(self.loss_function)
        # Iterate through the parameters in the signature
        arg_names = [param.name for param in sig.parameters.values()]
        return arg_names

    def __call__(self, *args, **kwargs):
        """
        Invokes the underlying loss function with the provided arguments.

        Returns:
            Any: Output of the loss function.

        Raises:
            LossError: If the loss function fails during execution.

        """
        try:
            return self.loss_function(*args, **kwargs)
        except Exception as e:
            raise LossError("Failed to call loss function.") from e

    def __repr__(self):
        if self.name:
            return f"Loss(name='{self.name}', backend='{self.backend.name}', reduction='{self.reduction}')"
        return f"Loss(custom_function={self.loss_function})"
