import warnings
from abc import ABC, abstractmethod

from modularml.components.shape_spec import ShapeSpec
from modularml.core.data.sample_schema import INVALID_LABEL_CHARACTERS
from modularml.utils.data_format import ensure_list
from modularml.utils.error_handling import ErrorMode
from modularml.utils.exceptions import GraphNodeInputError, GraphNodeOutputError


class GraphNode(ABC):
    """
    Abstract base class for all nodes within a ModelGraph.

    Each node is identified by a unique `label` and may have one or more upstream (input)
    and downstream (output) nodes. GraphNode defines the base interface for managing
    these connections and enforcing structural constraints like maximum allowed connections.

    Subclasses must define the `input_shape` and `output_shape` properties, as well as
    whether the node supports incoming and outgoing edges.
    """

    def __init__(
        self,
        label: str,
        upstream_nodes: str | list[str] | None = None,
        downstream_nodes: str | list[str] | None = None,
    ):
        """
        Initialize a GraphNode with optional upstream and downstream connections.

        Args:
            label (str): Unique identifier for this node.
            upstream_nodes (str | list[str] | None): Node labels to connect upstream.
            downstream_nodes (str | list[str] | None): Node labels to connect downstream.

        Raises:
            TypeError: If `upstream_nodes` or `downstream_nodes` are not valid types.

        """
        self._label = label

        # Normalize inputs as lists
        self._upstream_nodes = ensure_list(upstream_nodes)
        self._downstream_nodes = ensure_list(downstream_nodes)

        # Validate label and connections
        self._validate_label()
        self._validate_connections()

    def _validate_label(self):
        if any(ch in self.label for ch in INVALID_LABEL_CHARACTERS):
            msg = (
                f"The label contains invalid characters: `{self.label}`. "
                f"Label cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
            )
            raise ValueError(msg)

    def _validate_connections(self):
        # Enforce max_upstream_nodes
        if self.max_upstream_nodes is not None and len(self._upstream_nodes) > self.max_upstream_nodes:
            msg = f"{len(self._upstream_nodes)} upstream_nodes provided, but max_upstream_nodes = {self.max_upstream_nodes}"
            if self._handle_fatal_error(GraphNodeInputError, msg, ErrorMode.RAISE) is False:
                self._upstream_nodes = self._upstream_nodes[: self.max_upstream_nodes]

        # Enforce max_downstream_nodes
        if self.max_downstream_nodes is not None and len(self._downstream_nodes) > self.max_downstream_nodes:
            msg = f"{len(self._downstream_nodes)} downstream_nodes provided, but max_downstream_nodes = {self.max_downstream_nodes}"
            if self._handle_fatal_error(GraphNodeOutputError, msg, ErrorMode.RAISE) is False:
                self._downstream_nodes = self._downstream_nodes[: self.max_downstream_nodes]

    def __repr__(self):
        return f"GraphNode(label='{self.label}', upstream_nodes={self._upstream_nodes}, downstream_nodes={self._downstream_nodes})"

    def __str__(self):
        return f"GraphNode ('{self.label}')"

    @property
    def label(self) -> str:
        """Get or set the unique label for this node."""
        return self._label

    @label.setter
    def label(self, new_label: str):
        """Get or set the unique label for this node."""
        self._label = new_label
        self._validate_label()

    @property
    def max_upstream_nodes(self) -> int | None:
        """
        Maximum number of upstream (input) nodes allowed.

        Returns:
            int | None: If None, unlimited upstream nodes are allowed.

        """
        return None

    @property
    def max_downstream_nodes(self) -> int | None:
        """
        Maximum number of downstream (output) nodes allowed.

        Returns:
            int | None: If None, unlimited downstream nodes are allowed.

        """
        return None

    @property
    def upstream_node(self) -> str | None:
        """
        Return the single upstream node, if only one is allowed.

        Raises:
            RuntimeError: If multiple upstream nodes are allowed.

        """
        if self.max_upstream_nodes == 1:
            return self.get_upstream_nodes()[0] if self._upstream_nodes else None
        raise RuntimeError(
            "This node allows multiple upstream_nodes. Use `.get_upstream_nodes()`.",
        )

    @property
    def downstream_node(self) -> str | None:
        """
        Return the single downstream node, if only one is allowed.

        Raises:
            RuntimeError: If multiple downstream nodes are allowed.

        """
        if self.max_downstream_nodes == 1:
            return self.get_downstream_nodes()[0] if self._downstream_nodes else None
        raise RuntimeError(
            "This node allows multiple downstream_nodes. Use `.get_downstream_nodes()`.",
        )

    # ==========================================
    # Connection Management
    # ==========================================
    def get_upstream_nodes(self, error_mode: ErrorMode = ErrorMode.RAISE) -> list[str]:
        """
        Retrieve all upstream (input) nodes.

        Args:
            error_mode (ErrorMode): Error handling strategy if input is invalid.

        Returns:
            list[str]: List of node labels connected upstream.

        """
        if not self.allows_upstream_connections:
            handled = self._handle_benign_error(
                GraphNodeInputError,
                "This node does not allow upstream connections.",
                error_mode,
            )
            return [] if handled is False else self._upstream_nodes
        return self._upstream_nodes

    def get_downstream_nodes(
        self,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ) -> list[str]:
        """
        Retrieve all downstream (output) nodes.

        Args:
            error_mode (ErrorMode): Error handling strategy if input is invalid.

        Returns:
            list[str]: List of node labels connected downstream.

        """
        if not self.allows_downstream_connections:
            handled = self._handle_benign_error(
                GraphNodeOutputError,
                "This node does not allow downstream connections.",
                error_mode,
            )
            return [] if handled is False else self._downstream_nodes
        return self._downstream_nodes

    def add_upstream_node(self, node: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Add a single upstream connection.

        Args:
            node (str): Node label to connect upstream.
            error_mode (ErrorMode): Error handling mode for duplicates or limits.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_fatal_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            node in self._upstream_nodes
            and self._handle_benign_error(
                GraphNodeInputError,
                f"Upstream node '{node}' already connected.",
                error_mode,
            )
            is False
        ):
            return

        if (
            self.max_upstream_nodes is not None
            and len(self._upstream_nodes) > self.max_upstream_nodes
            and self._handle_fatal_error(
                GraphNodeInputError,
                f"Only {self.max_upstream_nodes} upstream_nodes allowed. Received: {self._upstream_nodes}",
                error_mode,
            )
            is False
        ):
            return

        self._upstream_nodes.append(node)

    def remove_upstream_node(self, node: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Remove a single upstream connection.

        Args:
            node (str): Node label to disconnect.
            error_mode (ErrorMode): Error handling mode if node not found.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_benign_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            node not in self._upstream_nodes
            and self._handle_benign_error(
                GraphNodeInputError,
                f"Upstream node '{node}' not connected.",
                error_mode,
            )
            is False
        ):
            return

        self._upstream_nodes.remove(node)

    def clear_upstream_nodes(self, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Remove all upstream connections.

        Args:
            error_mode (ErrorMode): Error handling mode if disallowed.

        """
        if (
            not self.allows_upstream_connections
            and self._handle_benign_error(
                GraphNodeInputError,
                "Upstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return
        self._upstream_nodes = []

    def set_upstream_nodes(
        self,
        upstream_nodes: list[str],
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Replace all upstream connections with a new list.

        Args:
            upstream_nodes (list[str]): List of new upstream node labels.
            error_mode (ErrorMode): Error handling mode for violations.

        """
        self.clear_upstream_nodes(error_mode=error_mode)
        for n in upstream_nodes:
            self.add_upstream_node(n, error_mode=error_mode)

    def add_downstream_node(self, node: str, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Add a single downstream connection.

        Args:
            node (str): Node label to connect downstream.
            error_mode (ErrorMode): Error handling mode for duplicates or limits.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_fatal_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            node in self._downstream_nodes
            and self._handle_benign_error(
                GraphNodeOutputError,
                f"Downstream node '{node}' already connected.",
                error_mode,
            )
            is False
        ):
            return

        if (
            self.max_downstream_nodes is not None
            and len(self._downstream_nodes) > self.max_downstream_nodes
            and self._handle_fatal_error(
                GraphNodeOutputError,
                f"Only {self.max_downstream_nodes} downstream_nodes allowed.",
                error_mode,
            )
            is False
        ):
            return

        self._downstream_nodes.append(node)

    def remove_downstream_node(
        self,
        node: str,
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Remove a single downstream connection.

        Args:
            node (str): Node label to disconnect.
            error_mode (ErrorMode): Error handling mode if node not found.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_benign_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return

        if (
            node not in self._downstream_nodes
            and self._handle_benign_error(
                GraphNodeOutputError,
                f"Downstream node '{node}' not connected.",
                error_mode,
            )
            is False
        ):
            return

        self._downstream_nodes.remove(node)

    def clear_downstream_nodes(self, error_mode: ErrorMode = ErrorMode.RAISE):
        """
        Remove all downstream connections.

        Args:
            error_mode (ErrorMode): Error handling mode if disallowed.

        """
        if (
            not self.allows_downstream_connections
            and self._handle_benign_error(
                GraphNodeOutputError,
                "Downstream connections not allowed.",
                error_mode,
            )
            is False
        ):
            return
        self._downstream_nodes = []

    def set_downstream_nodes(
        self,
        downstream_nodes: list[str],
        error_mode: ErrorMode = ErrorMode.RAISE,
    ):
        """
        Replace all downstream connections with a new list.

        Args:
            downstream_nodes (list[str]): List of new downstream node labels.
            error_mode (ErrorMode): Error handling mode for violations.

        """
        self.clear_downstream_nodes(error_mode=error_mode)
        for n in downstream_nodes:
            self.add_downstream_node(n, error_mode=error_mode)

    # ==========================================
    # Abstract Methods / Properties
    # ==========================================
    @property
    @abstractmethod
    def allows_upstream_connections(self) -> bool:
        """
        Whether this node allows incoming (upstream) connections.

        Returns:
            bool: True if input connections are allowed.

        """

    @property
    @abstractmethod
    def allows_downstream_connections(self) -> bool:
        """
        Whether this node allows outgoing (downstream) connections.

        Returns:
            bool: True if output connections are allowed.

        """

    @property
    @abstractmethod
    def input_shape_spec(self) -> ShapeSpec | None:
        """
        Shape of input data expected by this node.

        Returns:
            ShapeSpec | None: The expected input shape(s), or None if unknown.

        """

    @property
    @abstractmethod
    def output_shape_spec(self) -> ShapeSpec | None:
        """
        Shape of output data produced by this node.

        Returns:
            ShapeSpec | None: The expected output shape(s), or None if unknown.

        """

    def get_input_shape(self, key: str) -> tuple[int, ...]:
        if self.input_shape_spec is None:
            raise ValueError("Input ShapeSpec is None.")
        return self.input_shape_spec[key]

    def get_output_shape(self, key: str) -> tuple[int, ...]:
        if self.output_shape_spec is None:
            raise ValueError("Output ShapeSpec is None.")
        return self.output_shape_spec[key]

    # ==========================================
    # Internal Helpers
    # ==========================================
    def _handle_fatal_error(self, exc_class, message: str, error_mode: ErrorMode):
        """Raise or suppress a fatal error based on the provided error mode."""
        if error_mode == ErrorMode.IGNORE:
            return False
        if error_mode in (ErrorMode.WARN, ErrorMode.COERCE, ErrorMode.RAISE):
            raise exc_class(message)
        msg = f"Unsupported ErrorMode: {error_mode}"
        raise NotImplementedError(msg)

    def _handle_benign_error(self, exc_class, message: str, error_mode: ErrorMode):
        """Raise, warn, or ignore a non-fatal error based on the error mode."""
        if error_mode == ErrorMode.RAISE:
            raise exc_class(message)
        if error_mode == ErrorMode.WARN:
            warnings.warn(message, stacklevel=2, category=UserWarning)
            return False
        if error_mode in (ErrorMode.COERCE, ErrorMode.IGNORE):
            return False
        msg = f"Unsupported ErrorMode: {error_mode}"
        raise NotImplementedError(msg)
