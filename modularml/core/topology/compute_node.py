from abc import abstractmethod
from typing import Any

from modularml.core.data.batch import Batch, SampleData
from modularml.core.data.schema_constants import MML_STATE_TARGET
from modularml.core.references.reference_like import ReferenceLike
from modularml.core.topology.graph_node import GraphNode
from modularml.core.topology.node_shapes import NodeShapes


class ComputeNode(GraphNode):
    """
    Abstract base class for computational nodes in a ModelGraph.

    This class extends `GraphNode` and represents any node that performs tensor
    computation or transformation. It defines a contract for:
    - Shape inference and tracking
    - Forward computation (e.g., neural network layers or merge operations)
    - Backend-specific logic (e.g., PyTorch, TensorFlow)
    - Construction/build logic (e.g., instantiating models or computing shapes)

    Subclasses may include model stages, merge operations, or custom layers.
    """

    def __init__(
        self,
        label,
        upstream_refs: ReferenceLike | list[ReferenceLike] | None = None,
        downstream_refs: ReferenceLike | list[ReferenceLike] | None = None,
    ):
        """
        Initialize a ComputeNode.

        Args:
            label (str): Unique identifier for this node.
            upstream_refs (ReferenceLike | list[ReferenceLike] | None): References of upstream connections.
            downstream_refs (ReferenceLike | list[ReferenceLike] | None): References of downstream connections.

        """
        super().__init__(
            label,
            upstream_refs=upstream_refs,
            downstream_refs=downstream_refs,
        )

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("label", self.label),
            ("upstream_refs", [f"{r!r}" for r in self._upstream_refs]),
            ("downstream_refs", [f"{r!r}" for r in self._downstream_refs]),
            ("input_shapes", [(k, str(v)) for k, v in self.input_shapes.items()]),
            ("output_shapes", [(k, str(v)) for k, v in self.output_shapes.items()]),
        ]

    def __repr__(self):
        return (
            f"ComputeNode(label='{self.label}', "
            f"upstream_refs={self._upstream_refs}, "
            f"downstream_refs={self._downstream_refs})"
        )

    def __str__(self):
        return f"ComputeNode('{self.label}')"

    # ================================================
    # GraphNode Interface
    # ================================================
    @property
    def allows_upstream_connections(self) -> bool:
        """
        Whether this node allows incoming (upstream) connections.

        Returns:
            bool: True if input connections are allowed.

        """
        return True

    @property
    def allows_downstream_connections(self) -> bool:
        """
        Whether this node allows outgoing (downstream) connections.

        Returns:
            bool: True if output connections are allowed.

        """
        return True

    # ================================================
    # ComputeNode Interface
    # ================================================
    @abstractmethod
    def infer_output_shape(
        self,
        input_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        """
        Infer the output shapes of this node based on the given input shapes.

        Args:
            input_shapes (dict[str, tuple[int, ...]]):
                Input shapes feeding into this node.

        Returns:
            dict[str, tuple[int, ...]]: Inferred output shape(s).

        Raises:
            NotImplementedError: If the node cannot infer the shape without being built.

        """
        raise NotImplementedError

    @abstractmethod
    def get_input_data(self, batch: Batch) -> dict[str, SampleData] | list[dict[str, SampleData]]:
        """
        Retrieve and construct this node's input batch from all upstream batches.

        Args:
            batch (Batch):
                All Batch data for the ModelGraph forward pass.

        Returns:
            dict[str, SampleData] | list[dict[str, SampleData]]:
                Only the subset of :attr:`Batch.outputs` required for this node.

        """

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """
        Perform forward computation through the node.

        Args:
            inputs (Any): Input tensor(s), compatible with the backend (e.g., PyTorch, TensorFlow).

        Returns:
            Any: Output tensor(s) after computation.

        """

    @abstractmethod
    def build(
        self,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        output_shapes: dict[str, tuple[int, ...]] | None = None,
        **kwargs,
    ):
        """
        Construct the internal logic of this node using the provided input and output shapes.

        Args:
            input_shapes (dict[str, tuple[int, ...]] | None):
                Shapes of data feeding into this node.
                Used to initialize models or internal transformation logic.
            output_shapes (dict[str, tuple[int, ...]] | None):
                Shapes of data expected to exit this node.
                May be used to constrain or validate internal shape inference.
            **kwargs: Additional key-word arguments specific to each subclass.

        Notes:
            - Nodes with only a single input/output can simplify this logic.
            - This method should initialize any backend-specific model components.
            - If model or shape construction fails, this method should raise an error.

        """

    @property
    @abstractmethod
    def is_built(self) -> bool:
        """
        Whether this node has been fully built (e.g., model instantiated).

        Returns:
            bool: True if the node is fully built and ready for use.

        """

    @property
    @abstractmethod
    def node_shapes(self) -> NodeShapes:
        """Input and output shapes expected by this node."""

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Shape of input data expected by this node.

        Returns:
            dict[str, tuple[int, ...]]: The expected input shape(s).

        """
        return self.node_shapes.input_shapes

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Shape of output data expected by this node.

        Returns:
            dict[str, tuple[int, ...]]: The expected output shape(s).

        """
        return self.node_shapes.output_shapes

    # ================================================
    # Serialization
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Serialize GraphNode topology and identity.

        Includes:
            - ExperimentNode state (node_id, label)
            - upstream_refs
            - downstream_refs
        """
        state = super().get_state()
        state.update(
            {
                MML_STATE_TARGET: f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            },
        )
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore GraphNode identity and topology.

        Notes:
            - Parent (ExperimentNode) must be restored first
            - Reference existence is validated after restoration

        """
        # Restore node_id + label
        super().set_state(state)
