from abc import abstractmethod
from typing import Any

from modularml.core.data_structures.batch import Batch
from modularml.core.model_graph.graph_node import GraphNode


class ComputationNode(GraphNode):
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
        upstream_nodes: str | list[str] | None = None,
        downstream_nodes: str | list[str] | None = None,
    ):
        """
        Initialize a ComputationNode.

        Args:
            label (str): Unique identifier for this node.
            upstream_nodes (str | list[str] | None): Labels of upstream nodes.
            downstream_nodes (str | list[str] | None): Labels of downstream nodes.

        """
        super().__init__(label, upstream_nodes, downstream_nodes)

    def __repr__(self):
        return (
            f"ComputationNode(label='{self.label}', "
            f"upstream_nodes={self._upstream_nodes}, "
            f"downstream_nodes={self._downstream_nodes})"
        )

    def __str__(self):
        return f"ComputationNode ('{self.label}')"

    # ==========================================
    # GraphNode Interface
    # ==========================================
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

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Shape of input data expected by this node.

        Returns:
            tuple[int, ...] | None: The expected input shape, or None if unknown.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Shape of output data produced by this node.

        Returns:
            tuple[int, ...] | None: The output shape, or None if unknown.

        """

    # ==========================================
    # ComputationNode Interface
    # ==========================================
    @abstractmethod
    def infer_output_shapes(
        self,
        input_shapes: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """
        Infer the output shapes of this node based on the given input shapes.

        Args:
            input_shapes (list[tuple[int, ...]]): List of input shapes feeding into this node.
                I.e., a list of upstream node output shapes.

        Returns:
            list[tuple[int, ...]]: Inferred output shapes.

        Raises:
            NotImplementedError: If the node cannot infer the shape without being built.

        """
        raise NotImplementedError

    @abstractmethod
    def get_input_batch(self, all_batch_data: dict[str, Batch]) -> Batch:
        """
        Retrieve and construct this node's input batch from all upstream batches.

        Args:
            all_batch_data (dict[str, Batch]): Dictionary mapping node labels to Batch objects.

        Returns:
            Batch: The combined or transformed input batch for this node.

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
        input_shapes: list[tuple[int, ...]] | None = None,
        output_shapes: list[tuple[int, ...]] | None = None,
        **kwargs,
    ):
        """
        Construct the internal logic of this node using the provided input and output shapes.

        Args:
            input_shapes (list[tuple[int, ...]] | None): List of input tensor shapes.
                Used to initialize models or internal transformation logic.
            output_shapes (list[tuple[int, ...]] | None): Optional list of expected output shapes.
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
