

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import warnings
from modularml.utils.error_handling import ErrorMode
from modularml.utils.exceptions import GraphNodeInputError, GraphNodeOutputError


class GraphNode(ABC):
    """
    Abstract base class for any component that can be added to a ModelGraph.

    Each node has a label and tracks its upstream and downstream connections.
    Subclasses must define their input/output behavior and shape properties.
    """
    def __init__(
        self, 
        label: str, 
        inputs: Union[str, List[str]] = None, 
        outputs: Union[str, List[str]] = None, 
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        self._label = label
        
        # Normalize inputs to list
        if inputs is None:
            self._inputs: List[str] = []
        elif isinstance(inputs, str):
            self._inputs = [inputs]
        elif isinstance(inputs, list):
            self._inputs = inputs
        else:
            raise TypeError(f"`inputs` must be str, list[str], or None. Got: {type(inputs)}")

        # Check max_inputs
        if self.max_inputs is not None and len(self._inputs) > self.max_inputs:
            msg = f"{len(self._inputs)} input(s) provided, but max_inputs = {self.max_inputs}"
            if self._handle_error(GraphNodeInputError, msg, error_mode) is False:
                self._inputs = self._inputs[: self.max_inputs]

        # Normalize outputs to list
        if outputs is None:
            self._outputs: List[str] = []
        elif isinstance(outputs, str):
            self._outputs = [outputs]
        elif isinstance(outputs, list):
            self._outputs = outputs
        else:
            raise TypeError(f"`outputs` must be str, list[str], or None. Got: {type(outputs)}")        

        # Check max_outputs
        if self.max_outputs is not None and len(self._outputs) > self.max_outputs:
            msg = f"{len(self._outputs)} output(s) provided, but max_outputs = {self.max_outputs}"
            if self._handle_error(GraphNodeOutputError, msg, error_mode) is False:
                self._outputs = self._outputs[: self.max_outputs]

    @property
    def label(self) -> str:
        """The name assigned to this node."""
        return self._label
    
    @label.setter
    def label(self, new_label: str):
        """Set a new label for this node."""
        self._label = new_label
            
    @property
    def max_inputs(self) -> Optional[int]:
        """
        Maximum number of allowed input connections.
        - Return `None` for unlimited inputs (default).
        - Return an integer (e.g., 1) to enforce a limit.
        
        Child classes can overwrite to limit number of inputs.
        """
        return None
    
    @property
    def max_outputs(self) -> Optional[int]:
        """
        Maximum number of allowed output connections.
        - Return `None` for unlimited outputs (default).
        - Return an integer (e.g., 1) to enforce a limit.
        
        Child classes can overwrite to limit number of outputs.
        """
        return None

    @property
    def input(self) -> Optional[str]:
        if self.max_inputs == 1:
            return self.get_inputs()[0] if self._inputs else None
        raise RuntimeError("This node supports multiple inputs. Use `.get_inputs()` instead.")

    @property
    def output(self) -> Optional[str]:
        if self.max_outputs == 1:
            return self.get_outputs()[0] if self._outputs else None
        raise RuntimeError("This node supports multiple outputs. Use `.get_outputs()` instead.")



    def _handle_error(self, exc_class, message: str, error_mode: ErrorMode):
        if error_mode == ErrorMode.IGNORE:
            return False
        elif error_mode == ErrorMode.WARN:
            warnings.warn(message, category=UserWarning)
            return False
        elif error_mode == ErrorMode.COERCE:
            return False
        elif error_mode == ErrorMode.RAISE:
            raise exc_class(message)
        else:
            raise NotImplementedError(f"Unsupported ErrorMode: {error_mode}")
    
    def get_inputs(self, error_mode: ErrorMode = ErrorMode.RAISE) -> List[str]:
        """Get all input connections to this node."""
        if not self.allows_input_connections:
            handled = self._handle_error(GraphNodeInputError, "This node does not support input connections.", error_mode)
            return [] if handled is False else self._inputs
        return self._inputs
    
    def get_outputs(self, error_mode: ErrorMode = ErrorMode.RAISE) -> List[str]:
        """Get all output connections from this node."""
        if not self.allows_output_connections:
            handled = self._handle_error(GraphNodeOutputError, "This node does not support output connections.", error_mode)
            return [] if handled is False else self._outputs
        return self._outputs
    
    def add_input(self, input: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """Add a new input connection to this node."""
        if not self.allows_input_connections:
            if self._handle_error(GraphNodeInputError, "Inputs are not allowed on this GraphNode.", error_mode) is False:
                return

        if input in self._inputs:
            if self._handle_error(GraphNodeInputError, f"Input ({input}) already exists.", error_mode) is False:
                return

        # Enforce max_inputs constraint
        if self.max_inputs is not None and len(self._inputs) >= self.max_inputs:
            if self._handle_error(GraphNodeInputError, f"Only {self.max_inputs} input(s) allowed for this node.", error_mode) is False:
                return

        self._inputs.append(input)
      
    def remove_input(self, input: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """Remove an input connection from this node."""
        if not self.allows_input_connections:
            if self._handle_error(GraphNodeInputError, "Inputs are not allowed on this GraphNode.", error_mode) is False:
                return

        if input not in self._inputs:
            if self._handle_error(GraphNodeInputError, f"Input ({input}) does not exist.", error_mode) is False:
                return

        self._inputs.remove(input)
        
    def add_output(self, output: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """Add a new output connection from this node."""
        if not self.allows_output_connections:
            if self._handle_error(GraphNodeOutputError, "Outputs are not allowed on this GraphNode.", error_mode) is False:
                return

        if output in self._outputs:
            if self._handle_error(GraphNodeOutputError, f"Output ({output}) already exists.", error_mode) is False:
                return
            
        # Enforce max_outputs constraint
        if self.max_outputs is not None and len(self._inputs) >= self.max_outputs:
            if self._handle_error(GraphNodeInputError, f"Only {self.max_outputs} outputs(s) allowed for this node.", error_mode) is False:
                return

        self._outputs.append(output)

    def remove_output(self, output: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """Remove an output connection from this node."""
        if not self.allows_output_connections:
            if self._handle_error(GraphNodeOutputError, "Outputs are not allowed on this GraphNode.", error_mode) is False:
                return

        if output not in self._outputs:
            if self._handle_error(GraphNodeOutputError, f"Output ({output}) does not exist.", error_mode) is False:
                return

        self._outputs.remove(output)
    
    
    @property
    @abstractmethod
    def allows_input_connections(self) -> bool:
        """Whether this node type allows incoming connections."""
        pass
    
    @property
    @abstractmethod
    def allows_output_connections(self) -> bool:
        """Whether this node type allows outgoing connections."""
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """Returns the shape of the expected input to this node (if supports inputs)"""
        pass

    @property
    @abstractmethod
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Returns the shape of the output produced by this node (if supports outputs)"""
        pass
    
    