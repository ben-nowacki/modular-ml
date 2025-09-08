from collections.abc import Callable
from typing import Any

import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.exceptions import BackendNotSupportedError, OptimizerError, OptimizerNotSetError


class Optimizer:
    def __init__(self, name: str, backend: Backend, **optimizer_kwargs):
        """
        Initiallizes an Optimizer.

        Args:
            name (str): Name of the optimizer to use (e.g., "adam")
            backend (Backend): The backend to use (e.g., Backend.TORCH)
            **optimizer_kwargs: Additional keyword arguments to pass to optimizer

        """
        self.name = name.lower()
        self.backend = backend
        self.opt_cls = self._resolve()
        self.parameters = optimizer_kwargs.pop("model_parameters", None)
        self.opt_kwargs = optimizer_kwargs
        self.optimizer = None

    def _resolve(self) -> Callable:
        avail_opts = {}
        if self.backend == Backend.TORCH:
            avail_opts = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
                "rmsprop": torch.optim.RMSprop,
                "adagrad": torch.optim.Adagrad,
            }
        elif self.backend == Backend.TENSORFLOW:
            avail_opts = {
                "adam": tf.keras.optimizers.Adam,
                "adamw": tf.keras.optimizers.AdamW,
                "sgd": tf.keras.optimizers.SGD,
                "rmsprop": tf.keras.optimizers.RMSprop,
                "adagrad": tf.keras.optimizers.Adagrad,
            }
        else:
            raise BackendNotSupportedError(backend=self.backend, method="Optimizer._resolve()")

        opt = avail_opts.get(self.name)
        if opt is None:
            msg = (
                f"Unknown optimizer name (`{self.name}`) for `{self.backend}` backend."
                f"Available optimizers: {avail_opts.keys()}"
            )
            raise OptimizerError(msg)
        return opt

    def __call__(self, *, force_rebuild: bool = False, parameters: Any | None = None, **kwargs):
        """
        Instantiate the optimizer if not already provided.

        Args:
            force_rebuild (bool, optional): Whether to force rebuild optimizer.
            parameters (any, optional): Trainable parameters of the model (required for PyTorch).
            **kwargs: Kwargs to pass to optimizer

        """
        if not force_rebuild and self.is_built:
            raise OptimizerNotSetError(
                message=(
                    "Optimizer.__call__() is being called on an already instantiated optimizer. "
                    "If you want to rebuild the optimizer, set `force_rebuild=True`."
                ),
            )

        self.opt_kwargs |= kwargs

        if self.backend == Backend.TORCH:
            self.parameters = parameters
            if self.parameters is None:
                raise ValueError("Optimizer requires model parameters for the PyTorch backend.")
            self.optimizer = self.opt_cls(self.parameters, **self.opt_kwargs)

        elif self.backend == Backend.TENSORFLOW:
            self.optimizer = self.opt_cls(**self.opt_kwargs)

        else:
            raise BackendNotSupportedError(backend=self.backend, method="Optimzer.__call__()")

    def __repr__(self):
        msg_kwargs = ""
        for k, v in self.opt_kwargs.items():
            msg_kwargs += f", {k}={v}"
        return f"Optimizer('{self.name}'{msg_kwargs})"

    @property
    def is_built(self) -> bool:
        return self.optimizer is not None

    def build(self, *, force_rebuild: bool = False, parameters: Any | None = None, **kwargs):
        """
        Constructs the optimizer if not already provided.

        Args:
            force_rebuild (bool, optional): Whether to force rebuild optimizer.
            parameters (any, optional): Trainable parameters of the model (required for PyTorch).
            **kwargs: Kwargs to pass to optimizer

        """
        self.__call__(force_rebuild=force_rebuild, parameters=parameters, **kwargs)

    # ==========================================
    #   State/Config Management Methods
    # ==========================================
    def get_config(self) -> dict[str, Any]:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "backend": str(self.backend.value),
            "opt_kwargs": {
                **self.opt_kwargs,
                "model_parameters": self.parameters,
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Optimizer":
        return cls(name=config["name"], backend=Backend(config["backend"]), **config["opt_kwargs"])

    # ==========================================
    #   Error Checking Methods
    # ==========================================
    def _check_optimizer(self):
        if not self.is_built:
            raise OptimizerNotSetError(message="Optimizer has not been built.")

    # ==========================================
    #   Backpropagation Methods
    # ==========================================
    def step(self, grads=None, variables=None):
        """
        Perform optimizer step. For PyTorch, calls `optimizer.step()`.

        For TensorFlow, requires `grads` and `variables` and applies gradients.

        Args:
            grads: Gradients (required for TensorFlow).
            variables: Model variables (required for TensorFlow).

        """
        self._check_optimizer()

        if self.backend == Backend.TORCH:
            self.optimizer.step()

        elif self.backend == Backend.TENSORFLOW:
            if grads is None or variables is None:
                raise ValueError(
                    "Tensorflow backend requires both `grads` and `variables` to be set in Optimizer.step()"
                )
            self.optimizer.apply_gradients(zip(grads, variables, strict=True))

        else:
            raise BackendNotSupportedError(backend=self.backend, method="Optimizer.step()")

    def zero_grad(self):
        """Resets the optimizer gradients."""
        self._check_optimizer()

        if self.backend == Backend.TORCH:
            self.optimizer.zero_grad()

        elif self.backend == Backend.TENSORFLOW:
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))

        else:
            raise BackendNotSupportedError(backend=self.backend, method="Optimizer.zero_grad()")
