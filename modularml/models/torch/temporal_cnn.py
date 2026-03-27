"""Torch TemporalCNN reference implementation with lazy building."""

import numpy as np

from modularml.core.models.torch_base_model import TorchBaseModel
from modularml.utils.data.shape_utils import ensure_tuple_shape
from modularml.utils.data.types import TorchTensor
from modularml.utils.environment.optional_imports import check_torch
from modularml.utils.logging.warnings import warn
from modularml.utils.nn.activations import resolve_activation
from modularml.utils.nn.backend import Backend

torch = check_torch()


class ResidualBlock(torch.nn.Module):
    """
    One dilated residual block for use inside :class:`TemporalCNN`.

    Contains two weight-normalised dilated Conv1d layers with optional
    causal padding, configurable activation, and dropout.  A 1x1
    identity convolution is inserted on the skip path when the channel
    dimensions differ.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor for both conv layers.
        activation_name (str): Activation name forwarded to
            :func:`resolve_activation`.
        dropout (float): Dropout probability (applied after each
            conv-activation pair via :class:`torch.nn.Dropout1d`).
        causal (bool): When ``True``, uses left-only
            :class:`torch.nn.ConstantPad1d` so each output time-step
            only attends to past context.  When ``False``, symmetric
            padding is applied via ``Conv1d(padding=...)``.

    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation_name: str,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()

        pad = (kernel_size - 1) * dilation
        sym_pad = pad // 2

        # First dilated conv
        if causal:
            self.pad1 = torch.nn.ConstantPad1d((pad, 0), 0.0)
            self.conv1 = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                ),
            )
        else:
            self.pad1 = torch.nn.Identity()
            self.conv1 = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=sym_pad,
                    dilation=dilation,
                ),
            )
        self.act1 = resolve_activation(activation_name, backend=Backend.TORCH)
        self.dropout1 = (
            torch.nn.Dropout1d(p=dropout) if dropout > 0 else torch.nn.Identity()
        )

        # Second dilated conv
        if causal:
            self.pad2 = torch.nn.ConstantPad1d((pad, 0), 0.0)
            conv2 = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                ),
            )
        else:
            self.pad2 = torch.nn.Identity()
            conv2 = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=sym_pad,
                    dilation=dilation,
                ),
            )

        self.conv2 = conv2
        self.act2 = resolve_activation(activation_name, backend=Backend.TORCH)
        self.dropout2 = (
            torch.nn.Dropout1d(p=dropout) if dropout > 0 else torch.nn.Identity()
        )

        # Residual activation (applied after skip addition)
        self.act_residual = resolve_activation(activation_name, backend=Backend.TORCH)

        # 1x1 conv on the skip path when channel dims differ
        self.identity_conv = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: TorchTensor) -> TorchTensor:
        """
        Forward pass through the residual block.

        Args:
            x (TorchTensor): Input of shape ``(batch, in_channels, length)``.

        Returns:
            TorchTensor: Output of shape ``(batch, out_channels, length)``.

        """
        out = self.pad1(x)
        out = self.act1(self.conv1(out))
        out = self.dropout1(out)

        out = self.pad2(out)
        out = self.act2(self.conv2(out))
        out = self.dropout2(out)

        residual = x if self.identity_conv is None else self.identity_conv(x)
        return self.act_residual(out + residual)


class TemporalCNN(TorchBaseModel):
    """
    Temporal Convolutional Network.

    Each block contains two dilated Conv1d layers with weight
    normalization, configurable activation, and optional dropout.
    Dilation doubles per block, giving an exponentially growing
    receptive field.

    Attributes:
        n_blocks (int): Number of :class:`ResidualBlock` layers.
        hidden_dim (int): Channel width used throughout the network.
        kernel_size (int): Convolution kernel size.
        activation (str): Activation name resolved via
            :func:`resolve_activation`.
        dropout (float): Dropout probability applied after each
            conv-activation pair.
        causal (bool): Whether left-only causal padding is used.
        flatten_output (bool): Whether to append a linear projection
            to match the target output shape.
        tcn (torch.nn.Sequential | None): Stacked residual blocks.
        fc (torch.nn.Linear | None): Optional linear output projection.

    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        n_blocks: int = 4,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        causal: bool = True,
        flatten_output: bool = True,
        **kwargs,
    ):
        """
        Initialize the temporal CNN with optional lazy building.

        Args:
            input_shape (tuple[int, ...] | None): Shape excluding the
                batch dimension.
            output_shape (tuple[int, ...] | None): Desired output shape
                excluding the batch dimension.
            n_blocks (int): Number of residual blocks.  Dilation doubles
                per block (1, 2, 4, ...).
            hidden_dim (int): Number of channels in every residual block.
            kernel_size (int): Kernel size for all convolutions.
            activation (str): Activation name resolved via
                :func:`resolve_activation`.
            dropout (float): Dropout probability applied after each
                conv-activation pair inside each block.
            causal (bool): Use left-only causal padding when ``True``,
                symmetric padding when ``False``.
            flatten_output (bool): Append a linear projection to match
                ``output_shape`` when ``True``.
            **kwargs (Any): Extra keyword arguments forwarded to
                :class:`TorchBaseModel`.

        """
        # Pass all init args to parent for automatic get_config/from_config
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout,
            causal=causal,
            flatten_output=flatten_output,
            **kwargs,
        )

        # Ensure shape formatting (if given)
        self._input_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=input_shape,
            min_len=2,
            max_len=3,
            allow_null_shape=True,
        )
        self._output_shape: tuple[int, ...] | None = ensure_tuple_shape(
            shape=output_shape,
            min_len=2,
            max_len=3,
            allow_null_shape=True,
        )

        # Init args
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout = dropout
        self.causal = causal
        self.flatten_output = flatten_output

        # Layers (constructed in build())
        self.tcn = None
        self.fc = None

        # Build immediately if both shapes are known
        if self._input_shape and self._output_shape:
            self.build(self._input_shape, self._output_shape)

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Return the expected input shape excluding the batch dimension.

        Returns:
            tuple[int, ...] | None: Input shape when known.

        """
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Return the output shape excluding the batch dimension.

        Returns:
            tuple[int, ...] | None: Output shape when known.

        """
        return self._output_shape

    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build residual blocks and the optional linear projection.

        Args:
            input_shape (tuple[int, ...] | None):
                Shape excluding the batch dimension.
            output_shape (tuple[int, ...] | None):
                Target shape excluding the batch dimension.
            force (bool):
                Whether to rebuild existing layers.

        Raises:
            ValueError: If shapes conflict with previously supplied ones
                without forcing, or when ``input_shape`` is missing.
            UserWarning: If ``output_shape`` is missing and defaults are
                inferred.

        """
        # Validate / store input_shape
        if input_shape:
            input_shape = ensure_tuple_shape(
                shape=input_shape,
                min_len=2,
                max_len=None,
                allow_null_shape=False,
            )
            if (
                (self._input_shape is not None)
                and (input_shape != self._input_shape)
                and (not force)
            ):
                msg = (
                    f"Build called with `input_shape={input_shape}` but input shape is already defined "
                    f"with value `{self._input_shape}`. To override the existing shape, set `force=True`."
                )
                raise ValueError(msg)
            self._input_shape = input_shape

        # Validate / store output_shape
        if output_shape:
            output_shape = ensure_tuple_shape(
                shape=output_shape,
                min_len=2,
                max_len=None,
                allow_null_shape=False,
            )
            if (
                (self._output_shape is not None)
                and (output_shape != self._output_shape)
                and (not force)
            ):
                msg = (
                    f"Build called with `output_shape={output_shape}` but output shape is already defined "
                    f"with value `{self._output_shape}`. To override the existing shape, set `force=True`."
                )
                raise ValueError(msg)
            self._output_shape = output_shape

        if self.is_built and not force:
            return

        if self._input_shape is None:
            raise ValueError("Input shape must be provided before building the model.")
        if self._output_shape is None:
            msg = "No output shape provided. Using default shape of (1, hidden_dim)."
            warn(msg, category=UserWarning, stacklevel=2)
            self._output_shape = (1, self.hidden_dim)

        # Stack n_blocks residual blocks; dilation doubles per block
        in_channels = self._input_shape[0]
        blocks = []
        for i in range(self.n_blocks):
            dilation = 2**i
            block_in = in_channels if i == 0 else self.hidden_dim
            blocks.append(
                ResidualBlock(
                    in_channels=block_in,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    activation_name=self.activation,
                    dropout=self.dropout,
                    causal=self.causal,
                ),
            )

        self.tcn = torch.nn.Sequential(*blocks)

        if self.flatten_output:
            dummy = torch.zeros(1, *self._input_shape)
            with torch.no_grad():
                tcn_out = self.tcn(dummy)
            conv_out_dim = tcn_out.shape[1] * tcn_out.shape[2]

            flat_output_dim = int(np.prod(self._output_shape))
            self.fc = torch.nn.Linear(conv_out_dim, flat_output_dim)

        self._built = True

    def forward(self, x: TorchTensor) -> TorchTensor:
        """
        Run the forward pass through the temporal CNN.

        Args:
            x (TorchTensor):
                Input tensor of shape ``(batch, channels, length)``.
                A 2-D input ``(batch, length)`` has a channel dimension
                inserted automatically.

        Returns:
            TorchTensor: Output tensor shaped ``(batch, *output_shape)``.

        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)

        if not self.is_built:
            self.build(input_shape=tuple(x.shape[1:]))

        x = self.tcn(x)

        if self.flatten_output:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), *self._output_shape)

        return x
