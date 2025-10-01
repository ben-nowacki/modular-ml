from collections.abc import Callable
from typing import Any

import tensorflow as tf
import torch

from modularml.utils.backend import Backend
from modularml.utils.exceptions import BackendNotSupportedError, OptimizerError, OptimizerNotSetError


class Optimizer:
    """
    Backend-agnostic optimizer wrapper.

    Description:
        Encapsulates optimizer creation logic with support for lazy construction,
        backend inference, and config-based serialization.

    Initialization Options:
        - From backend and name: `Optimizer(name="adam", lr=1e-3)`
        - From class + kwargs:   `Optimizer(cls=torch.optim.Adam, lr=...)`
    """

    def __init__(
        self,
        name: str | None = None,
        cls: type | None = None,
        backend: Backend | None = None,
        **kwargs,
    ):
        if name is not None and cls is not None:
            msg = (
                "Optimizer can be set with either an optimzier name (eg, 'relu') "
                "or a class (eg, `toch.optim.Adam`), but not both."
            )
            raise ValueError(msg)

        self.cls = None
        self.kwargs = kwargs
        if name is not None:
            self.name = name.lower()
            self._backend = backend
            if self._backend is not None:
                self.cls = self._resolve()  # set optimizer cls
        elif cls is not None:
            self.cls = cls

        self.instance = None
        self._backend = backend  # If not provided, inferred later

    def __repr__(self):
        msg_kwargs = ""
        for k, v in self.kwargs.items():
            msg_kwargs += f", {k}={v}"
        return f"Optimizer('{self.name}'{msg_kwargs})"

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

    @property
    def is_built(self) -> bool:
        return self.instance is not None

    @property
    def backend(self) -> Backend:
        return self._backend

    @backend.setter
    def backend(self, value: Backend):
        self._backend = value

    def build(
        self,
        *,
        force_rebuild: bool = False,
        parameters: Any | None = None,
        backend: Backend | None = None,
        **kwargs,
    ):
        """
        Constructs the optimizer if not already provided.

        Args:
            force_rebuild (bool, optional): Whether to force rebuild optimizer.
            parameters (any, optional): Trainable parameters of the model (required for PyTorch).
            backend (Backend | None): Backend to use for optimizer, if not already set.
            **kwargs: Kwargs to pass to optimizer

        """
        self.__call__(
            force_rebuild=force_rebuild,
            parameters=parameters,
            backend=backend,
            **kwargs,
        )

    def __call__(
        self,
        *,
        force_rebuild: bool = False,
        parameters: Any | None = None,
        backend: Backend | None = None,
        **kwargs,
    ):
        """
        Instantiate the optimizer if not already provided.

        Args:
            force_rebuild (bool, optional): Whether to force rebuild optimizer.
            parameters (any, optional): Trainable parameters of the model (required for PyTorch).
            backend (Backend | None): Backend to use for optimizer, if not already set.
            **kwargs: Kwargs to pass to optimizer

        """
        if not force_rebuild and self.is_built:
            raise OptimizerNotSetError(
                message=(
                    "Optimizer.__call__() is being called on an already instantiated optimizer. "
                    "If you want to rebuild the optimizer, set `force_rebuild=True`."
                ),
            )

        # Set backend if not already set
        if backend is not None and self._backend is not None:
            if backend != self._backend:
                msg = f"Backend passed to Optimizer.build differs from backend at init: {backend} != {self._backend}"
                raise ValueError(msg)
        elif self._backend is None:
            self._backend = backend
        if self._backend is None:
            raise ValueError("Backend must be provided during init or build.")

        # Set optimizer class if not already set
        if self.cls is None:
            self.cls = self._resolve()

        self.kwargs |= kwargs

        if self._backend == Backend.TORCH:
            self.parameters = parameters
            if self.parameters is None:
                raise ValueError("Optimizer requires model parameters for the PyTorch backend.")
            self.instance = self.cls(self.parameters, **self.kwargs)

        elif self._backend == Backend.TENSORFLOW:
            self.instance = self.cls(**self.kwargs)

        else:
            raise BackendNotSupportedError(backend=self._backend, method="Optimzer.__call__()")

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

        if self._backend == Backend.TORCH:
            self.instance.step()

        elif self._backend == Backend.TENSORFLOW:
            if grads is None or variables is None:
                raise ValueError(
                    "Tensorflow backend requires both `grads` and `variables` to be set in Optimizer.step()",
                )
            self.instance.apply_gradients(zip(grads, variables, strict=True))

        else:
            raise BackendNotSupportedError(backend=self._backend, method="Optimizer.step()")

    def zero_grad(self):
        """Resets the optimizer gradients."""
        self._check_optimizer()

        if self._backend == Backend.TORCH:
            self.instance.zero_grad()

        elif self._backend == Backend.TENSORFLOW:
            for var in self.instance.variables():
                var.assign(tf.zeros_like(var))

        else:
            raise BackendNotSupportedError(backend=self._backend, method="Optimizer.zero_grad()")

    # # ==========================================
    # #   State/Config Management Methods
    # # ==========================================
    # def get_config(self) -> dict[str, Any]:
    #     return {
    #         "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
    #         "name": self.name,
    #         "backend": str(self.backend.value),
    #         "opt_kwargs": {
    #             **self.opt_kwargs,
    #             "model_parameters": self.parameters,
    #         },
    #     }

    # @classmethod
    # def from_config(cls, config: dict[str, Any]) -> "Optimizer":
    #     return cls(name=config["name"], backend=Backend(config["backend"]), **config["opt_kwargs"])
