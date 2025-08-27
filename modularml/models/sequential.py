
import warnings
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


from modularml.models.base import BaseModel
from modularml.utils.backend import Backend
from modularml.core.model_graph.activation import Activation


class SequentialMLP(BaseModel, torch.nn.Module):
    def __init__(
        self, 
        input_shape: Optional[Tuple[int]] = None,
        output_shape: Optional[Tuple[int]] = None,
        n_layers: int = 2,
        hidden_dim: int = 32,
        activation: str = 'relu',
        dropout: float = 0.0,
        backend: Optional[Backend] = Backend.TORCH
    ):
        """
        Initializes a sequential MLP model using PyTorch backend.
        Actual model layers are built in the `build()` method.

        Args:
            input_shape (Tuple[int], optional): Shape of input sample (n_features, feature_len)
            output_shape (Tuple[int], optional): Desired shape of output
            n_layers (int): Number of fully-connected layers
            hidden_dim (int): Size of hidden units
            activation (str): Activation function name
            dropout (float): Dropout rate
            backend (Backend): ML backend (default is PyTorch)
        """
        config = {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "dropout": dropout,
        }
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, config=config, backend=backend)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc = None          # Will be created in build()

        # Build immediately if shape is specified
        if self.input_shape is not None and self.output_shape is not None:
            self.build()

    def build(self, input_shape: Optional[Tuple[int]] = None, output_shape: Optional[Tuple[int]] = None):
        """
        Constructs the internal sequential layers. Called lazily when input/output shape is known.
        
        Args:
            input_shape (Optional[Tuple[int]], optional): Used if model not built (num_features, feature_len). \
                Defaults to None.
            output_shape (Optional[Tuple[int]], optional): Used if model not built (num_targets, target_len). \
                Defaults to None.
        """
        if self.is_built(): return
        
        # Set input and output shapes
        if input_shape is not None:
            if self.input_shape is not None and not self.input_shape == input_shape:
                raise ValueError(
                    f"A new input_shape ({input_shape}) was provided that doesn't match the previous ({self.input_shape})"
                )
            self.input_shape = input_shape
        if self.input_shape is None:
            raise ValueError("Input shape must be provided to build the model.")
        
        if output_shape is not None:
            if self.output_shape is not None and not self.output_shape == output_shape:
                raise ValueError(
                    f"A new output_shape ({output_shape}) was provided that doesn't match the previous ({self.output_shape})"
                )
            self.output_shape = output_shape
        if self.output_shape is None:
            warnings.warn(
                f"No output shape was provided. The shape of (1, `hidden_dim`) will be used.",
                category=UserWarning,
                stacklevel=2
            )
            self.output_shape = (1, self.config['hidden_dim'])
            
        # Instantiate activation
        act_fnc = Activation(name=self.config["activation"], backend=self.backend)

        # Get flat dim sizes
        flat_input_dim = int(np.prod(self.input_shape))
        flat_output_dim = int(np.prod(self.output_shape))

        layers = []
        for i in range(self.config["n_layers"] - 1):
            in_dim = flat_input_dim if i == 0 else self.config["hidden_dim"]
            layers.append(torch.nn.Linear(
                in_features=in_dim, 
                out_features=self.config["hidden_dim"]
            ))
            layers.append(act_fnc.get_layer())
            if self.config["dropout"] > 0:
                layers.append(torch.nn.Dropout(self.config["dropout"]))

        final_input = self.config["hidden_dim"] if self.config["n_layers"] > 1 else flat_input_dim
        layers.append(torch.nn.Linear(final_input, flat_output_dim))

        self.fc = torch.nn.Sequential(*layers)
        self.mark_built()

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_features, feature_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)
        """
        if not self.is_built():
            self.build(input_shape=tuple(x.shape[1:]))  # input_shape = (n_features, feature_len)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), *self.output_shape)

    def __call__(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_features, feature_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)
        """
        return self.forward(x)


class SequentialCNN(BaseModel, torch.nn.Module):
    def __init__(
        self, 
        input_shape: Optional[Tuple[int]] = None,
        output_shape: Optional[Tuple[int]] = None,
        n_layers: int = 2,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        activation: str = 'relu',
        dropout: float = 0.0,
        pooling: int = 1,
        flatten_output: bool = True,
        backend: Backend = Backend.TORCH,
    ):
        """
        Initializes a sequential CNN with lazy layer building.

        Args:
            input_shape (Tuple[int], optional): Shape (n_features, feature_len)
            output_shape (Tuple[int], optional): Desired final shape
            n_layers (int): Number of CNN layers
            hidden_dim (int): Output size of each layer
            kernel_size (int): Kernel size for convolution
            padding (int): Padding size
            stride (int): Stride length
            activation (str): Activation function
            dropout (float): Dropout probability
            pooling (int): Pooling kernel size (1 = no pooling)
            flatten_output (bool): Whether to flatten output and apply linear layer
            backend (Backend): Backend (default: torch)
        """
        config = {
            "input_shape": input_shape,
            "output_shape": output_shape,
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
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, config=config, backend=backend)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.flatten = flatten_output

        self.conv_layers = None  # Built in build()
        self.fc = None           # Final linear layer if flatten=True

        # Build immediately if shape is specified
        if self.input_shape is not None and self.output_shape is not None:
            self.build()

    def build(self, input_shape: Optional[Tuple[int]] = None, output_shape: Optional[Tuple[int]] = None):
        """
        Builds convolutional layers and final FC layer if needed.

        Args:
            input_shape (Optional[Tuple[int]], optional): Used if model not built (num_features, feature_len). \
                Defaults to None.
            output_shape (Optional[Tuple[int]], optional): Used if model not built (num_targets, target_len). \
                Defaults to None.
        """
        if self.is_built(): return
        
        # Set input and output shapes
        if input_shape is not None:
            if self.input_shape is not None and not self.input_shape == input_shape:
                raise ValueError(
                    f"A new input_shape ({input_shape}) was provided that doesn't match the previous ({self.input_shape})"
                )
            self.input_shape = input_shape
        if self.input_shape is None:
            raise ValueError("Input shape must be provided to build the model.")
        
        if output_shape is not None:
            if self.output_shape is not None and not self.output_shape == output_shape:
                raise ValueError(
                    f"A new output_shape ({output_shape}) was provided that doesn't match the previous ({self.output_shape})"
                )
            self.output_shape = output_shape
        if self.output_shape is None:
            warnings.warn(
                f"No output shape was provided. The shape of (1, `hidden_dim`) will be used.",
                category=UserWarning,
                stacklevel=2
            )
            self.output_shape = (1, self.config['hidden_dim'])
            
        # Instantiate activation
        act_fnc = Activation(name=self.config["activation"], backend=self.backend)

        num_features, feature_len = input_shape
        layers = []
        for _ in range(self.config["n_layers"]):
            layers.append(torch.nn.Conv1d(
                in_channels=num_features,
                out_channels=self.config["hidden_dim"],
                kernel_size=self.config["kernel_size"],
                stride=self.config["stride"],
                padding=self.config["padding"],
            ))
            layers.append(act_fnc.get_layer())

            if self.config["dropout"] > 0:
                layers.append(torch.nn.Dropout(self.config["dropout"]))

            if self.config["pooling"] > 1 and feature_len >= self.config["pooling"]:
                layers.append(torch.nn.MaxPool1d(kernel_size=self.config["pooling"]))
                feature_len = feature_len // self.config["pooling"]

            num_features = self.config["hidden_dim"]

        self.conv_layers = torch.nn.Sequential(*layers)

        if self.flatten:
            dummy = torch.zeros(1, *input_shape)
            with torch.no_grad():
                conv_out = self.conv_layers(dummy)
            conv_out_dim = conv_out.shape[1] * conv_out.shape[2]

            if self.output_shape is None:
                self.output_shape = (conv_out_dim,)  # default fallback

            output_dim = int(np.prod(self.output_shape))
            self.fc = torch.nn.Linear(conv_out_dim, output_dim)

        self.mark_built()

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Sample with shape (batch_size, n_features, feature_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)
        """
        if not self.is_built():
            self.build(input_shape=x.shape[1:])  # input_shape = (n_features, feature_len)
        
        x = self.conv_layers(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), *self.output_shape)
        return x

    def __call__(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): Sample with shape (batch_size, n_features, feature_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *output_shape)
        """
        return self.forward(x)

