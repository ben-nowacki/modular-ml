from typing import Any

from modularml.core.data.sample_data import SampleShapes


class NodeShapes:
    """
    Input and output tensor shape specification for a GraphNode.

    Summary:
        `NodeShapes` describes the expected tensor shapes flowing *into* and
        *out of* a GraphNode during model graph execution. Shapes represent
        model-level tensors (e.g., features flowing between nodes), and are
        stored as tuples of integers excluding the batch dimension.

    """

    _INVALID_PREFIXES = ("input_", "output_")

    def __init__(
        self,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        output_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        """
        Input and output tensor shape specification for a GraphNode.

        Args:
            input_shapes (dict[str, tuple[int, ...]] | None):
                Mapping of input port names to tensor shapes. Defaults to `{}`.

            output_shapes (dict[str, tuple[int, ...]] | None):
                Mapping of output port names to tensor shapes. Defaults to `{}`.

        """
        self.input_shapes = {str(k): v for k, v in (input_shapes or {}).items()}
        self.output_shapes = {str(k): v for k, v in (output_shapes or {}).items()}

        self._validate_port_names(self.input_shapes, kind="input")
        self._validate_port_names(self.output_shapes, kind="output")

        # Validate shapes are tuples of ints
        if not all(self._is_valid_shape(v) for v in self.input_shapes.values()):
            raise ValueError("Input shape values must be tuple[int, ...] entries.")
        if not all(self._is_valid_shape(v) for v in self.output_shapes.values()):
            raise ValueError("Output shape values must be tuple[int, ...] entries.")

    def __repr__(self):
        return f"NodeShapes(inputs={self.input_shapes}, outputs={self.output_shapes})"

    # ==========================================
    # Naming utilities
    # ==========================================
    @staticmethod
    def make_input_name(port_name: str) -> str:
        """Return the public name for an input port."""
        return f"input_{port_name}"

    @staticmethod
    def make_output_name(port_name: str) -> str:
        """Return the public name for an output port."""
        return f"output_{port_name}"

    # ==========================================
    # Shape accessors
    # ==========================================
    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Return all shapes as a flat dict with canonical prefixed keys.

        Example:
            {
                "input_0": (32, 128),
                "input_1": (32, 64),
                "output_0": (32, 192),
            }

        """
        out = {}

        for pn, shape in self.input_shapes.items():
            out[self.make_input_name(pn)] = shape

        for pn, shape in self.output_shapes.items():
            out[self.make_output_name(pn)] = shape

        return out

    # ==========================================
    # Helpers and validation
    # ==========================================
    @staticmethod
    def _is_valid_shape(shape: Any) -> bool:
        """Validate that a shape is a tuple of ints."""
        return isinstance(shape, tuple) and all(isinstance(d, int) for d in shape)

    def _validate_port_names(self, mapping: dict[str, SampleShapes], kind: str):
        """Ensure that no port name uses disallowed prefixes."""
        for name in mapping:
            if any(name.startswith(prefix) for prefix in self._INVALID_PREFIXES):
                msg = (
                    f"Invalid port name '{name}' for {kind}_shapes. Port names cannot begin with 'input_' or 'output_'."
                )
                raise ValueError(msg)
