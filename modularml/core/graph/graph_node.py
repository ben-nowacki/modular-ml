import warnings
from abc import ABC, abstractmethod

from modularml.core.graph.shape_spec import ShapeSpec
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

        # Normalize upstream_nodes
        if upstream_nodes is None:
            self._upstream_nodes: list[str] = []
        elif isinstance(upstream_nodes, str):
            self._upstream_nodes = [upstream_nodes]
        elif isinstance(upstream_nodes, list):
            self._upstream_nodes = upstream_nodes
        else:
            msg = f"`upstream_nodes` must be str, list[str], or None. Got: {type(upstream_nodes)}"
            raise TypeError(msg)

        # Enforce max_upstream_nodes
        if self.max_upstream_nodes is not None and len(self._upstream_nodes) > self.max_upstream_nodes:
            msg = f"{len(self._upstream_nodes)} upstream_nodes provided, but max_upstream_nodes = {self.max_upstream_nodes}"
            if self._handle_fatal_error(GraphNodeInputError, msg, ErrorMode.RAISE) is False:
                self._upstream_nodes = self._upstream_nodes[: self.max_upstream_nodes]

        # Normalize downstream_nodes
        if downstream_nodes is None:
            self._downstream_nodes: list[str] = []
        elif isinstance(downstream_nodes, str):
            self._downstream_nodes = [downstream_nodes]
        elif isinstance(downstream_nodes, list):
            self._downstream_nodes = downstream_nodes
        else:
            msg = f"`downstream_nodes` must be str, list[str], or None. Got: {type(downstream_nodes)}"
            raise TypeError(msg)

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
