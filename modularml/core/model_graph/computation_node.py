



from abc import abstractmethod
from typing import Any, Optional, Tuple
from modularml.core.model_graph.graph_node import GraphNode
from modularml.utils.backend import Backend



class ComputationNode(GraphNode):
    """
    Abstract base class for nodes that perform computations in a ModelGraph.

    This class extends `GraphNode` and provides an interface for model stages 
    or tensor operations that require shape inference, execution (`forward`), 
    and backend-specific behavior (e.g., PyTorch, TensorFlow).
    """
    
    def __init__(self, label, inputs = None, outputs = None):
        """
        Initialize a ComputationNode.

        Args:
            label (str): The unique label/name for this node.
            inputs (Union[str, List[str]], optional): Names of upstream nodes.
            outputs (Union[str, List[str]], optional): Names of downstream nodes.
        """
        super().__init__(label, inputs, outputs)
        
    # ==========================================
    # GraphNode Methods
    # ==========================================
    @property
    def allows_input_connections(self) -> bool:
        """
        Indicates that this node supports input connections.

        Returns:
            bool: Always True for ComputationNode.
        """
        return True
    @property
    def allows_output_connections(self) -> bool:
        """
        Indicates that this node supports output connections.

        Returns:
            bool: Always True for ComputationNode.
        """
        return True
    
    @property
    @abstractmethod
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Shape of the expected input to this node.

        Returns:
            Optional[Tuple[int, ...]]: Input shape tuple (e.g., (32, 64)) or None if unknown.
        """
        pass
    
    @property
    @abstractmethod
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Shape of the output produced by this node.

        Returns:
            Optional[Tuple[int, ...]]: Output shape tuple (e.g., (32, 10)) or None if unknown.
        """
        pass
    
    
    
    # ==========================================
    # ComputationNode Specific Methods
    # ==========================================
    @property
    @abstractmethod
    def backend(self) -> Backend:
        """
        Backend used by the underlying model or computation.

        Returns:
            Backend: The backend enum (e.g., Backend.TORCH, Backend.TENSORFLOW).
        """
        pass

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """
        Perform a forward pass through this node using backend-specific logic.

        Args:
            inputs (Any): Input tensor(s) in the appropriate format for the backend.

        Returns:
            Any: Output tensor(s) after forward computation.
        """
        pass

    @abstractmethod
    def infer_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Infer the output shape based on the provided input shape.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input tensor.

        Returns:
            Tuple[int, ...]: Inferred output shape.
        """
        pass
    
    @abstractmethod
    def build(self, input_shape: Optional[Tuple[int]] = None, output_shape: Optional[Tuple[int]] = None):
        """
        Build any backend-specific model components or internal properties.

        Args:
            input_shape (Optional[Tuple[int, ...]]): Input shape, if known.
            output_shape (Optional[Tuple[int, ...]]): Output shape, if known.

        Notes:
            If shape is not known, this method may raise an error or defer building.
        """
        pass

    @property
    @abstractmethod
    def is_built(self) -> bool:
        """
        Whether this node has been fully built (e.g., model instantiated).

        Returns:
            bool: True if built, False otherwise.
        """
        pass

