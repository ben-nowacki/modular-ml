from typing import Any

import numpy as np
import tensorflow as tf
import torch

from modularml.core.graph.graph_node import GraphNode
from modularml.utils.backend import Backend
from modularml.utils.data_conversion import convert_to_format
from modularml.utils.data_format import get_data_format_for_backend
from modularml.utils.modeling import PadMode, map_pad_mode_to_backend

from .merge_stage import MergeStage


class ConcatStage(MergeStage):
    """
    A merge stage that concatenates multiple inputs along a specified axis.

    Description:
        This stage merges tensors by concatenating them along a specified axis. It supports
        automatic padding of non-concat dimensions to align shapes, allowing for flexible
        merging even when inputs vary in size. Padding behavior can be controlled via mode
        (e.g., 'constant', 'reflect', 'replicate', 'circular') and value.

    Attributes:
        label (str): Unique identifier for this node.
            upstream_nodes (list[str | GraphNode]): Optional list of upstream
            node labels or references from which inputs will be received.
        axis (int): The axis along which to concatenate inputs.
        pad_inputs (bool, optional): Whether to pad inputs before merging. Defaults to False.
        pad_mode (PadMode, optional): Padding mode ('constant', 'reflect', 'replicate', etc.). Defaults to 'constant'.
        pad_value (float, optional): Value to use for constant padding. Defaults to 0.0.
        **kwargs: Additional arguments passed to `MergeStage`.

    Example:
    ```python
        stage = ConcatStage(label="merge", upstreadm_nodes=[...], axis=1, pad_inputs=True, pad_mode="constant", pad_value=0.0)
    ```

    """

    def __init__(
        self,
        label: str,
        upstream_nodes: list[str | GraphNode],
        axis: int = 0,
        *,
        pad_inputs: bool = False,
        pad_mode: PadMode = PadMode.CONSTANT,
        pad_value: float = 0.0,
        **kwargs,
    ):
        """
        Initialize a ConcatStage node.

        Args:
            label (str): Unique identifier for this node.
            upstream_nodes (list[str | GraphNode]): Optional list of upstream
                node labels or references from which inputs will be received.
            axis (int): The axis along which to concatenate inputs.
            pad_inputs (bool, optional): Whether to pad inputs before merging. Defaults to False.
            pad_mode (PadMode, optional): Padding mode ('constant', 'reflect', 'replicate', etc.). Defaults to 'constant'.
            pad_value (float, optional): Value to use for constant padding. Defaults to 0.0.
            **kwargs: Additional arguments passed to `MergeStage`.

        """
        super().__init__(label=label, upstream_nodes=upstream_nodes, **kwargs)
        self.concat_axis = axis
        self.pad_inputs = pad_inputs
        self.pad_mode = pad_mode if isinstance(pad_mode, PadMode) else PadMode(pad_mode)
        self.pad_value = pad_value

        if self.pad_mode not in [PadMode.CONSTANT]:
            msg = f"Pad mode is not supported yet: {self.pad_mode}"
            raise NotImplementedError(msg)

    def _pad_inputs(self, values: list[Any]) -> list[Any]:
        """
        Pad all inputs along non-concat dimensions to match the largest shape.

        Description:
            This method applies backend-specific padding logic to ensure that all tensors
            have the same shape (except for the concat axis) before concatenation.

        Args:
            values (list[Any]): List of tensors to be padded.

        Returns:
            list[Any]: Padded tensors.

        Raises:
            ValueError: If the backend is unsupported or if padding fails.

        """
        # Determine max shape along each axis
        max_shape = np.max([np.array(v.shape) for v in values], axis=0)
        padded = []
        for v in values:
            pad_width = []
            for dim, current_shape in enumerate(v.shape):
                if dim == self.concat_axis:
                    pad_width.append((0, 0))  # No padding on concat axis
                else:
                    diff = max_shape[dim] - current_shape
                    pad_width.append((0, diff))

            # Apply backend-specific pad function (to ensure no breakage of gradients)
            if self._backend == Backend.TORCH:
                torch_pad = [p for dims in reversed(pad_width) for p in dims]  # reverse & flatten
                p = torch.nn.functional.pad(
                    input=v,
                    pad=torch_pad,
                    mode=map_pad_mode_to_backend(mode=self.pad_mode, backend=self._backend),
                    value=self.pad_value,
                )
                padded.append(p)
            elif self._backend == Backend.TENSORFLOW:
                tf_pad = tf.constant(pad_width)
                p = tf.pad(
                    tensor=v,
                    paddings=tf_pad,
                    mode=map_pad_mode_to_backend(mode=self.pad_mode, backend=self._backend),
                    constant_values=self.pad_value,
                )
                padded.append(p)
            elif self._backend == Backend.SCIKIT:
                p = np.pad(
                    array=v,
                    pad_width=pad_width,
                    mode=map_pad_mode_to_backend(mode=self.pad_mode, backend=self._backend),
                    constant_values=self.pad_value,
                )
                padded.append(p)
            else:
                msg = f"Unsupported backend for padding: {self._backend}"
                raise ValueError(msg)

        return padded

    def _validate_dims(self, values: list[Any], concat_axis: int):
        reference_shape = values[0].shape
        for i, v in enumerate(values[1:], start=1):
            for dim, (ref_dim, val_dim) in enumerate(zip(reference_shape, v.shape, strict=True)):
                if dim == concat_axis:
                    continue
                if ref_dim != val_dim:
                    msg = (
                        f"Mismatch in non-concat dimension {dim} between input 0 and {i}: "
                        f"{ref_dim} vs {val_dim}. Set `pad_inputs=True` to auto-align. "
                        f"{reference_shape} vs {v.shape} on axis={concat_axis}"
                    )
                    raise ValueError(msg)

    def apply_merge(self, values: list[Any], *, includes_batch_dim: bool = True) -> Any:
        """
        Concatenate input tensors along the specified axis.

        Description:
            Optionally pads the inputs to align non-concat dimensions before applying
            backend-specific concatenation.

        Args:
            values (list[Any]): List of input tensors to be merged.
            includes_batch_dim (bool): Whether the inputs values have a batch dimension. \
                Defaults to True.

        Returns:
            Any: Concatenated tensor, in the backend-specific format.

        Raises:
            ValueError: If the backend is unsupported.

        """
        # Ensure all elements in list are converted to the appropriate backend
        values = [convert_to_format(x, fmt=get_data_format_for_backend(self._backend)) for x in values]

        # Apply padding if defined
        if self.pad_inputs:
            values = self._pad_inputs(values)

        else:
            self._validate_dims(values=values, concat_axis=self.concat_axis + int(includes_batch_dim))

        # Apply backend-specific concatentation function (to ensure no breakage of gradients)
        if self._backend == Backend.TORCH:
            return torch.concatenate(tensors=values, dim=self.concat_axis + int(includes_batch_dim))
        if self._backend == Backend.TENSORFLOW:
            return tf.concat(values=values, axis=self.concat_axis + int(includes_batch_dim))
        if self._backend == Backend.SCIKIT:
            return np.concatenate(values, axis=self.concat_axis + int(includes_batch_dim))

        msg = f"Unsupported backend for concatenation: {self._backend}"
        raise ValueError(msg)
