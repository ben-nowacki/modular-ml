import warnings

import numpy as np
import torch

from modularml.core.activation.activation import Activation
from modularml.models.base import BaseModel
from modularml.utils.backend import Backend


class SequentialMLP(BaseModel, torch.nn.Module):
    """
    Configurable multi-layer perceptron (MLP) with lazy shape inference.

    This model wraps a PyTorch sequential MLP, with optional layer count, activation,
    and dropout. It supports lazy building when input/output shapes are unknown.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int] | None = None,
        output_shape: tuple[int] | None = None,
        n_layers: int = 2,
        hidden_dim: int = 32,
        activation: str = "relu",
        dropout: float = 0.0,
        backend: Backend | None = Backend.TORCH,
    ):
        """
        Configurable multi-layer perceptron (MLP) with lazy shape inference.

        This model wraps a PyTorch sequential MLP, with optional layer count, activation,
        and dropout. It supports lazy building when input/output shapes are unknown.

        Args:
            input_shape (Tuple[int], optional): Shape of the input excluding batch dim.
            output_shape (Tuple[int], optional): Shape of the output excluding batch dim.
            n_layers (int): Number of fully connected layers.
            hidden_dim (int): Number of hidden units per layer.
            activation (str): Activation function name (e.g., 'relu', 'gelu').
            dropout (float): Dropout rate (0.0 = no dropout).
            backend (Backend): ML backend to use (must be Backend.TORCH).

        """
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, backend=backend)

        self._input_shape = input_shape
        self._output_shape = output_shape

        self.config = {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "dropout": dropout,
        }
        self.fc = None  # Will be set in build()

        if self._input_shape and self._output_shape:  # build immediately if given shapes
            self.build(self._input_shape, self._output_shape)

    @property
    def input_shape(self) -> tuple[int] | None:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int] | None:
        return self._output_shape

    def build(
        self,
        input_shape: tuple[int] | None = None,
        output_shape: tuple[int] | None = None,
    ):
        """
        Builds the internal torch.nn.Sequential model.

        Args:
            input_shape (Tuple[int], optional): Input shape excluding batch dim.
            output_shape (Tuple[int], optional): Output shape excluding batch dim.

        Raises:
            ValueError: If shape mismatch is detected.

        """
        if input_shape:
            if self._input_shape and input_shape != self._input_shape:
                msg = f"Inconsistent input_shape: {input_shape} vs {self._input_shape}"
                raise ValueError(msg)
            self._input_shape = input_shape

        if output_shape:
            if self._output_shape and output_shape != self._output_shape:
                msg = f"Inconsistent output_shape: {output_shape} vs {self._output_shape}"
                raise ValueError(msg)
            self._output_shape = output_shape

        if self.is_built:
            return

        if self._input_shape is None:
            raise ValueError("Input shape must be provided before building the model.")
        if self._output_shape is None:
            warnings.warn(
                "No output shape provided. Using default shape of (1, hidden_dim).",
                stacklevel=2,
                category=UserWarning,
            )
            self._output_shape = (1, self.config["hidden_dim"])

        flat_input = int(np.prod(self._input_shape))
        flat_output = int(np.prod(self._output_shape))
        act_fn = Activation(self.config["activation"], backend=self.backend).get_layer()

        layers = []
        for i in range(self.config["n_layers"] - 1):
            in_dim = flat_input if i == 0 else self.config["hidden_dim"]
            layers.append(torch.nn.Linear(in_dim, self.config["hidden_dim"]))
            layers.append(act_fn)
            if self.config["dropout"] > 0:
                layers.append(torch.nn.Dropout(self.config["dropout"]))

        final_in = self.config["hidden_dim"] if self.config["n_layers"] > 1 else flat_input
        layers.append(torch.nn.Linear(final_in, flat_output))

        self.fc = torch.nn.Sequential(*layers)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input of shape (batch_size, *input_shape)

        Returns:
            torch.Tensor: Output of shape (batch_size, *output_shape)

        """
        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), *self._output_shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
