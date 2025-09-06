from typing import Any

import numpy as np
import tensorflow as tf
import torch

from modularml.core.graph.merge_stage import MergeStage
from modularml.utils.backend import Backend
from modularml.utils.data_format import convert_to_format, get_data_format_for_backend


class ConcatStage(MergeStage):
    """
    A merge stage that concatenates multiple inputs along a specified axis.

    Description:
        This stage merges tensors by concatenating them along a specified axis. It supports
        automatic padding of non-concat dimensions to align shapes, allowing for flexible
        merging even when inputs vary in size. Padding behavior can be controlled via mode
        (e.g., 'constant', 'reflect', 'replicate', 'circular') and value.

    Attributes:
        concat_axis (int): The axis along which to concatenate inputs.
        pad_inputs (bool): Whether to pad inputs to align shapes before concatenation.
        pad_mode (str): Padding mode to use. Options depend on backend.
        pad_value (float): Constant value to use when `pad_mode='constant'`.

    Example:
        >>> stage = ConcatStage(label="merge", axis=1, pad_inputs=True, pad_mode="constant", pad_value=0.0)

    """

    def __init__(
        self,
        label: str,
        axis: int = -1,
        *,
        pad_inputs: bool = False,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        **kwargs,
    ):
        """
        Initialize a ConcatStage node.

        Args:
            label (str): Unique identifier for this node.
            axis (int): The axis along which to concatenate inputs.
            pad_inputs (bool, optional): Whether to pad inputs before merging. Defaults to False.
            pad_mode (str, optional): Padding mode ('constant', 'reflect', 'replicate', etc.). Defaults to 'constant'.
            pad_value (float, optional): Value to use for constant padding. Defaults to 0.0.
            **kwargs: Additional arguments passed to `MergeStage`.

        """
        super().__init__(label=label, **kwargs)
        self.concat_axis = axis
        self.pad_inputs = pad_inputs
        self.pad_mode = pad_mode.lower()
        self.pad_value = pad_value

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
                padded.append(torch.nn.functional.pad(v, torch_pad, mode=self.pad_mode, value=self.pad_value))
            elif self._backend == Backend.TENSORFLOW:
                tf_pad = tf.constant(pad_width)
                padded.append(tf.pad(v, tf_pad, mode=self.pad_mode.upper(), constant_values=self.pad_value))
            elif self._backend == Backend.SCIKIT:
                padded.append(np.pad(v, pad_width, mode=self.pad_mode, constant_values=self.pad_value))
            else:
                msg = f"Unsupported backend for padding: {self._backend}"
                raise ValueError(msg)

        return padded

    def apply_merge(self, values: list[Any]) -> Any:
        """
        Concatenate input tensors along the specified axis.

        Description:
            Optionally pads the inputs to align non-concat dimensions before applying
            backend-specific concatenation.

        Args:
            values (list[Any]): List of input tensors to be merged.

        Returns:
            Any: Concatenated tensor, in the backend-specific format.

        Raises:
            ValueError: If the backend is unsupported.

        """
        # Ensure all elements in list are converted to the appropriate backend
        values = [convert_to_format(x, format=get_data_format_for_backend(self._backend)) for x in values]

        # Apply padding if defined
        if self.pad_inputs:
            values = self._pad_inputs(values)

        # Apply backend-specific concatentation function (to ensure no breakage of gradients)
        if self._backend == Backend.TORCH:
            return torch.concatenate(tensors=values, dim=self.concat_axis)
        if self._backend == Backend.TENSORFLOW:
            return tf.concat(values=values, axis=self.concat_axis)
        if self._backend == Backend.SCIKIT:
            return np.concatenate(values, axis=self.concat_axis)
        msg = f"Unsupported backend for concatenation: {self._backend}"
        raise ValueError(msg)
