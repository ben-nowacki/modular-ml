from abc import ABC, abstractmethod
from typing import Any, overload

import numpy as np

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.data import Data
from modularml.core.model_graph.computation_node import ComputationNode
from modularml.core.model_graph.graph_node import GraphNode
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_to_format, get_data_format_for_backend


class MergeStage(ComputationNode, ABC):
    """
    Base class for merging multiple upstream nodes in a model graph.

    Description:
        A MergeStage represents a node in the model graph that takes multiple upstream
        nodes and merges their outputs into a single output. This class serves as an abstract
        base for concrete merging strategies such as concatenation, averaging, or summation.

        Subclasses must implement the `apply_merge()` method, which defines how values
        from multiple inputs are combined into a single tensor or structure.

    Examples:
        >>> class MyConcatStage(MergeStage):
        >>>     def apply_merge(self, values): return np.concatenate(values, axis=1)

    """

    def __init__(
        self,
        label: str,
        upstream_nodes: list[str | GraphNode] | None = None,
    ):
        """
        Initialize a MergeStage.

        Args:
            label (str): Unique identifier for this node.
            upstream_nodes (list[str | GraphNode] | None): Optional list of upstream
                node labels or references from which inputs will be received.

        """
        super().__init__(label=label, upstream_nodes=upstream_nodes)
        self._merged_shape: tuple[int, ...] | None = None
        self._built = False
        self._backend = Backend.NONE

    # ==========================================
    # ComputationNode Interface
    # ==========================================
    @property
    def is_built(self) -> bool:
        """
        Whether the MergeStage has been built (i.e., shape inference completed).

        Returns:
            bool: True if built, False otherwise.

        """
        return self._built

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Accessing input_shape is not supported for MergeStage.

        Raises:
            ValueError: Always raised to indicate this property is invalid for merges.

        """
        msg = (
            "MergeStages do not have a single input shape. "
            "Use output_shape or merged_shape to access the final merged shape."
        )
        raise ValueError(msg)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Returns the final output shape of the merged result.

        Description:
            For merge operations, the output shape is equivalent to the merged shape.
            This property can be used by downstream nodes to infer their input shapes.

        Returns:
            tuple[int, ...]: Merged output shape (excluding batch dimension).

        """
        return self.merged_shape

    @property
    def merged_shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the merged output.

        Description:
            This is the final shape (excluding batch dimension) after combining inputs
            from all upstream nodes.

        Returns:
            tuple[int, ...]: Merged output shape.

        Raises:
            RuntimeError: If the node is not yet built or the merged shape is not set.

        """
        if self._merged_shape is None:
            if self.is_built:
                raise RuntimeError("merged_shape is None after MergeStage building.")
            raise RuntimeError("MergeStage must be built before accessing merged_shape")
        return self._merged_shape

    @abstractmethod
    def apply_merge(self, values: list[Any]) -> Any:
        """
        Merge logic to be implemented by subclasses.

        Args:
            values (list[Any]): A list of backend-specific tensors to be merged.

        Returns:
            Any: Merged tensor in backend-native format.

        """

    def infer_output_shapes(
        self,
        input_shapes: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """
        Infer output shapes based on input shapes from upstream nodes.

        Args:
            input_shapes (list[tuple[int, ...]]): Shapes from upstream outputs.

        Returns:
            list[tuple[int, ...]]: A list with the inferred output shape.

        Raises:
            RuntimeError: If backend is not yet set.

        """
        if self._backend == Backend.NONE:
            msg = f"Backend not set in MergeStage `{self.label}`. Make sure `.build()` was called."
            raise RuntimeError(msg)
        merged_data = self.apply_merge(values=[np.ones(shape=x) for x in input_shapes])
        if isinstance(merged_data, list):
            return [x.shape for x in merged_data]
        return [convert_to_format(merged_data, format=DataFormat.NUMPY).shape]

    def build(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]] | None = None,
        *,
        backend: Backend,
    ):
        """
        Build the MergeStage and perform shape inference.

        Args:
            input_shapes (list[tuple[int, ...]]): Shapes from upstream outputs.
            output_shapes (list[tuple[int, ...]] | None): Not supported.
            backend (Backend): Backend to use for merging (e.g., TORCH, TENSORFLOW).

        Raises:
            ValueError: If input_shapes is missing or output_shapes is provided.

        """
        self._backend = backend

        if input_shapes is None:
            msg = f"MergeStage `{self.label}` requires input_shapes during build."
            raise ValueError(msg)

        if output_shapes is not None:
            msg = f"MergeStage `{self.label}` does not accept output_shapes during build. Received: {output_shapes}"

        # Determine merged_shape
        inferred = self.infer_output_shapes(input_shapes=input_shapes)
        if len(inferred) > 1:
            msg = f"infer_output_shape of MergeStage `{self.label}` resulted in more than one output_shape"
            raise ValueError(msg)

        self._merged_shape = inferred[0]
        self._built = True

    def forward(
        self,
        all_inputs: list[Data] | list[Batch | BatchOutput],
    ) -> Data | BatchOutput:
        """
        Merge multiple inputs (Data, Batch, or BatchOutput) into a single output.

        Args:
            all_inputs (list): List of inputs to merge.

        Returns:
            Union[Data, BatchOutput]: Merged output matching input type.

        Raises:
            TypeError: If input is not a list or has mixed/unsupported types.
            RuntimeError: If backend is not yet set.
            ValueError: If roles across batches are not aligned.

        """
        if not isinstance(all_inputs, list):
            msg = f"MergeStage inputs must be a list of input data. Received: {type(all_inputs)}"
            raise TypeError(msg)
        if self._backend == Backend.NONE:
            msg = f"Backend not set in MergeStage `{self.label}`. Make sure `.build()` was called."
            raise RuntimeError(msg)

        # Case 1: all inputs are Data
        if all(isinstance(x, Data) for x in all_inputs):
            data_to_merge = [x.value for x in all_inputs]
            merged = self.apply_merge(values=data_to_merge)
            return Data(value=merged)

        # Case 2: all inputs are Batch or BatchOutput
        if all(isinstance(x, Batch | BatchOutput) for x in all_inputs):
            # To merge multiple Batches (or BatchOutputs), all roles must match
            list_of_roles = [set(x.available_roles) for x in all_inputs]
            if len({frozenset(s) for s in list_of_roles}) != 1:
                msg = f"Cannot merge batches with different roles. Detected roles per input: {list_of_roles}"
                raise ValueError(msg)

            # Merge inputs for each role
            merged_features: dict[str, Any] = {}
            merged_targets: dict[str, Any] = {}
            merged_uuids: dict[str, list[str]] = {}
            roles = list_of_roles[0]
            for r in roles:
                features = []
                targets = []
                sample_uuids = []
                for x in all_inputs:
                    if isinstance(x, Batch):
                        features.append(
                            x.get_samples(role=r).get_all_features(format=get_data_format_for_backend(self._backend)),
                        )
                        targets.append(
                            x.get_samples(role=r).get_all_targets(format=get_data_format_for_backend(self._backend)),
                        )
                        sample_uuids.append(
                            x.get_samples(role=r).sample_uuids,
                        )
                    elif isinstance(x, BatchOutput):
                        features.append(
                            convert_to_format(
                                x.features[r],
                                format=get_data_format_for_backend(self._backend),
                            )  # noqa: COM812
                        )
                        targets.append(
                            convert_to_format(
                                x.targets[r],
                                format=get_data_format_for_backend(self._backend),
                            )  # noqa: COM812
                        )
                        sample_uuids.append(x.sample_uuids[r])

                merged_targets[r] = self.apply_merge(values=targets)
                merged_features[r] = self.apply_merge(values=features)
                merged_uuids[r] = list(zip(*sample_uuids, strict=True))

            return BatchOutput(
                features=merged_features,
                sample_uuids=merged_uuids,
                targets=merged_targets,
            )

        msg = f"Combination of input types is not supported in MergeStage: {[type(x) for x in all_inputs]}"
        raise TypeError(msg)

    # ==========================================
    # MergeStage Properties & Dunders
    # ==========================================
    def __repr__(self):
        return f"{self!s}"

    def __str__(self):
        return f"MergeStage ('{self.label}')"

    @overload
    def __call__(self, all_inputs: list[Data], **kwargs) -> Data: ...
    @overload
    def __call__(self, all_inputs: list[Batch | BatchOutput], **kwargs) -> BatchOutput: ...
    def __call__(self, all_inputs: list[Data] | list[Batch | BatchOutput], **kwargs) -> Data | BatchOutput:
        """
        Shorthand for `.forward()`.

        Args:
            all_inputs (list): Inputs to merge.
            **kwargs: Passed through to `.forward()`.

        Returns:
            Data | BatchOutput: Merged result.

        """
        return self.forward(all_inputs=all_inputs, **kwargs)
