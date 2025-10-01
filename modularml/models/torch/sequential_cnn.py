import warnings

import numpy as np
import torch

from modularml.core.activation.activation import Activation
from modularml.models.base import BaseModel
from modularml.utils.backend import Backend


class SequentialCNN(BaseModel, torch.nn.Module):
    """
    Configurable 1D convolutional neural network (CNN) with lazy shape inference.

    This model supports stacked Conv1D layers followed optionally by dropout, pooling,
    and a final flattening linear projection to a user-defined output shape.
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int] | None = None,
        output_shape: tuple[int] | None = None,
        n_layers: int = 2,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        pooling: int = 1,
        flatten_output: bool = True,
        backend: Backend = Backend.TORCH,
    ):
        """
        Initialize a sequential 1D CNN model with lazy layer building.

        Args:
            input_shape (tuple[int] | None): Input tensor shape (channels, length).
            output_shape (tuple[int] | None): Desired output shape (after flattening if enabled).
            n_layers (int): Number of stacked convolutional layers.
            hidden_dim (int): Number of output channels for each Conv1D layer.
            kernel_size (int): Kernel size for convolution.
            padding (int): Padding for convolution.
            stride (int): Stride for convolution.
            activation (str): Activation function name (e.g., 'relu', 'tanh').
            dropout (float): Dropout probability (0.0 = no dropout).
            pooling (int): Pooling kernel size (1 = no pooling).
            flatten_output (bool): If True, appends a flattening linear projection to match output shape.
            backend (Backend): Backend framework to use (default: Backend.TORCH).

        """
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, backend=backend)

        self._input_shape = input_shape
        self._output_shape = output_shape

        self.config = {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "kernel_size": kernel_size,
            "padding": padding,
            "stride": stride,
            "activation": activation,
            "dropout": dropout,
            "pooling": pooling,
            "flatten_output": flatten_output,
        }

        self.conv_layers = None
        self.fc = None

        if self._input_shape and self._output_shape:
            self.build(self._input_shape, self._output_shape)

    @property
    def input_shape(self) -> tuple[int] | None:
        """Returns the expected input shape (channels, length) excluding batch dimension."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int] | None:
        """Returns the output shape of the model excluding batch dimension."""
        return self._output_shape

    def build(
        self,
        input_shape: tuple[int] | None = None,
        output_shape: tuple[int] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build the CNN layers and optional final linear projection.

        Args:
            input_shape (tuple[int] | None): Input shape (channels, length) to override initial input shape.
            output_shape (tuple[int] | None): Output shape to override initial output shape.
            force (bool): If model is already instantiated, force determines whether to reinstantiate with \
                the new shapes. Defaults to False.

        Raises:
            ValueError: If inconsistent shape is passed compared to earlier ones, or if input shape is not specified.
            UserWarning: If output shape is not provided, a default fallback is used.

        """
        if input_shape:
            if self._input_shape and input_shape != self._input_shape and not force:
                msg = f"Inconsistent input_shape: {input_shape} vs {self._input_shape}"
                raise ValueError(msg)
            self._input_shape = input_shape

        if output_shape:
            if self._output_shape and output_shape != self._output_shape and not force:
                msg = f"Inconsistent output_shape: {output_shape} vs {self._output_shape}"
                raise ValueError(msg)
            self._output_shape = output_shape

        if self.is_built and not force:
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

        act_fn = Activation(self.config["activation"], backend=self.backend).get_layer()

        num_features, feature_len = self._input_shape
        layers = []
        for _ in range(self.config["n_layers"]):
            layers.append(
                torch.nn.Conv1d(
                    in_channels=num_features,
                    out_channels=self.config["hidden_dim"],
                    kernel_size=self.config["kernel_size"],
                    stride=self.config["stride"],
                    padding=self.config["padding"],
                ),
            )
            layers.append(act_fn)

            if self.config["dropout"] > 0:
                layers.append(torch.nn.Dropout(self.config["dropout"]))

            if self.config["pooling"] > 1 and feature_len >= self.config["pooling"]:
                layers.append(torch.nn.MaxPool1d(kernel_size=self.config["pooling"]))
                feature_len = feature_len // self.config["pooling"]

            num_features = self.config["hidden_dim"]

        self.conv_layers = torch.nn.Sequential(*layers)

        if self.config["flatten_output"]:
            dummy = torch.zeros(1, *self._input_shape)
            with torch.no_grad():
                conv_out = self.conv_layers(dummy)
            conv_out_dim = conv_out.shape[1] * conv_out.shape[2]

            if self._output_shape is None:
                self._output_shape = (conv_out_dim,)

            flat_output_dim = int(np.prod(self._output_shape))
            self.fc = torch.nn.Linear(conv_out_dim, flat_output_dim)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)

        """
        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))

        x = self.conv_layers(x)
        if self.config["flatten_output"]:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), *self._output_shape)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward(x).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)

        """
        return self.forward(x)
