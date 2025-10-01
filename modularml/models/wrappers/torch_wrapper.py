import inspect
import warnings
from collections.abc import Callable
from typing import Any

import torch

from modularml.models.base import BaseModel
from modularml.utils.backend import Backend


class TorchModelWrapper(BaseModel, torch.nn.Module):
    """
    A wrapper for PyTorch models to conform to the ModularML BaseModel interface.

    This class supports two construction modes:
    - Eager mode: Wrapping an already-instantiated `torch.nn.Module`
    - Lazy mode: Delaying construction by passing a model class and optional keyword arguments

    When using lazy mode, input/output shapes can be inferred during the `build()` step and
    optionally injected into the model's constructor under user-defined keywords.
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        model_class: Callable | None = None,
        model_kwargs: dict[str, Any] | None = None,
        inject_input_shape_as: str = "input_shape",
        inject_output_shape_as: str = "output_shape",
    ):
        """
        Initialize the TorchModelWrapper.

        Args:
            model (torch.nn.Module, optional): An already-instantiated PyTorch model.
            model_class (Callable, optional): A PyTorch model class (e.g., `MyModelClass`).
            model_kwargs (dict, optional): Keyword arguments to construct the model. \
                May omit `input_shape` and/or `output_shape` for lazy inference.
            inject_input_shape_as (str): If provided, `input_shape` will be injected into \
                `model_class` under this keyword, unless it already exists in `model_kwargs`.
            inject_output_shape_as (str): If provided, `output_shape` will be injected into \
                `model_class` under this keyword, unless it already exists in `model_kwargs`.

        Raises:
            ValueError: If neither `model` nor `model_class` is provided, or if input types are invalid.

        """
        super().__init__(backend=Backend.TORCH)

        self.model = model
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self._inject_input_shape_as = inject_input_shape_as
        self._inject_output_shape_as = inject_output_shape_as

        self._input_shape = None
        self._output_shape = None

        # Check validity
        if self.model is not None:
            if not isinstance(self.model, torch.nn.Module):
                msg = "Provided model must be an instance of torch.nn.Module."
                raise ValueError(msg)
        elif self.model_class is not None:
            if not callable(self.model_class):
                msg = "model_class must be callable."
                raise ValueError(msg)
        else:
            msg = "Must provide either `model` or (`model_class` + `model_kwargs`)."
            raise ValueError(msg)

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """Returns the input shape (if known)."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        """Returns the output shape (if known)."""
        return self._output_shape

    def _validate_model_shapes(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        """
        Validates that the wrapped PyTorch model matches expected input and output shapes.

        This method performs a dummy forward pass using a random tensor of shape `(1, *input_shape)`.
        If the model fails to run or the output shape does not match `output_shape` (if provided),
        a RuntimeError is raised.

        Args:
            input_shape (tuple[int, ...] | None): Expected input shape, excluding batch dimension.
            output_shape (tuple[int, ...] | None): Optional expected output shape to verify.

        Raises:
            RuntimeError: If the model fails to accept the input shape or if the output
                shape does not match the expected shape.

        """
        dummy_in = torch.randn(1, *input_shape)

        # Check that pre-instantiated model accepts expected input_shape
        try:
            with torch.no_grad():
                dummy_out = self.model(dummy_in)
        except Exception as e:
            msg = f"Pre-instantiated PyTorch model failed to accept the expected input_shape: {input_shape}"
            raise RuntimeError(msg) from e

        # Check that pre-instantiated model returns expected output_shape
        if output_shape is not None and dummy_out.shape[1:] != output_shape:
            msg = (
                "Output shape of pre-instantiated PyTorch model does not match expected output_shape: "
                f" {dummy_out.shape[1:]} != {output_shape}"
            )
            raise RuntimeError(msg)

    def _get_model_init_kwargs(
        self,
        input_shape: tuple[int, ...] | None,
        output_shape: tuple[int, ...] | None,
    ) -> dict[str, Any]:
        kwargs = dict(self.model_kwargs)

        # Inspect model_class constructor
        sig = inspect.signature(self.model_class.__init__)
        accepted_params = sig.parameters.keys()

        # Try to inject input shape
        if self._inject_input_shape_as not in kwargs and input_shape is not None:
            # If accepted keyword, just inject
            if self._inject_input_shape_as in accepted_params:
                kwargs[self._inject_input_shape_as] = input_shape

            # User explicitly set inject_input_shape_as but not accepted -> raise error
            elif self._inject_input_shape_as != "input_shape":
                msg = f"`{self._inject_input_shape_as}` is not accepted by `{self.model_class.__name__}` constructor."
                raise ValueError(msg)

            # User never changed inject_input_shape_as but default not accepted -> warn & skip injection
            else:
                # Default injection failed → warn
                warnings.warn(
                    f"Model constructor for `{self.model_class.__name__}` does not accept "
                    f"`input_shape` as a kwarg. Skipping injection.",
                    stacklevel=2,
                    category=RuntimeWarning,
                )

        # Try to inject output shape
        if self._inject_output_shape_as not in kwargs and output_shape is not None:
            # If accepted keyword, just inject
            if self._inject_output_shape_as in accepted_params:
                kwargs[self._inject_output_shape_as] = output_shape

            # User explicitly set inject_output_shape_as but not accepted -> raise error
            elif self._inject_output_shape_as != "output_shape":
                msg = f"`{self._inject_output_shape_as}` is not accepted by `{self.model_class.__name__}` constructor."
                raise ValueError(msg)

            # User never changed inject_output_shape_as but default not accepted -> warn & skip injection
            else:
                # Default injection failed → warn
                warnings.warn(
                    f"Model constructor for `{self.model_class.__name__}` does not accept "
                    f"`output_shape` as a kwarg. Skipping injection.",
                    stacklevel=2,
                    category=RuntimeWarning,
                )

        return kwargs

    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,  # noqa: ARG002
    ):
        """
        Construct or validate the wrapped PyTorch model.

        If a model instance was provided at initialization, this method validates its input
        and output shapes using a dummy forward pass. If a model class and keyword arguments
        were provided instead, the model is instantiated by injecting the inferred input and/or
        output shape if not already present in the provided kwargs.

        This method ensures the model is compatible with the ModularML shape propagation logic
        and supports both eager and lazy model creation strategies.

        Args:
            input_shape (tuple[int, ...] | None): Input shape (excluding batch dimension) to use
                for validation or lazy instantiation.
            output_shape (tuple[int, ...] | None): Expected output shape (excluding batch dimension)
                to use for validation or lazy instantiation.
            force (bool): If model is already instantiated, force determines whether \
                to reinstantiate with the new shapes. Defaults to False.

        Raises:
            RuntimeError: If validation fails for a pre-instantiated model, or if model construction
                fails due to invalid arguments or shape mismatches.

        """
        # Case 1: Model was pre-instantiated
        if self.model is not None:
            # Already built, validate that inputs + outputs match
            self._validate_model_shapes(input_shape=input_shape, output_shape=output_shape)
            self._input_shape = input_shape
            self._output_shape = output_shape
            self._built = True
            return

        # Case 2: Model class + kwargs provided
        if self.model_class is not None:
            # Inject input/output shape if not already provided
            kwargs = self._get_model_init_kwargs(input_shape=input_shape, output_shape=output_shape)

            # Attempt model instantiation
            try:
                self.model = self.model_class(**kwargs)
            except Exception as e:
                msg = f"Failed to instantiate PyTorch model with args: {kwargs}"
                raise RuntimeError(msg) from e

            self._validate_model_shapes(input_shape=input_shape, output_shape=output_shape)
            self._input_shape = input_shape
            self._output_shape = output_shape
        else:
            msg = "No model_class or model provided to build from."
            raise RuntimeError(msg)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output from the wrapped model.

        """
        return self.model(x)

    def __call__(self, *args, **kwargs):
        """Alias to forward pass."""
        return self.forward(*args, **kwargs)
