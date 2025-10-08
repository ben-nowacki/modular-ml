from __future__ import annotations

import copy
import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import tensorflow as tf

from modularml.core.data_structures.step_result import StepResult
from modularml.core.graph.computation_node import ComputationNode
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.graph_node import GraphNode
from modularml.core.graph.merge_stages.merge_stage import MergeStage
from modularml.core.graph.mixins import EvaluableMixin, TrainableMixin
from modularml.core.graph.model_stage import ModelStage
from modularml.core.graph.shape_spec import ShapeSpec
from modularml.core.loss.loss_collection import LossCollection
from modularml.core.loss.loss_record import LossRecord
from modularml.utils.backend import Backend, backend_requires_optimizer
from modularml.utils.error_handling import ErrorMode
from modularml.utils.exceptions import BackendNotSupportedError
from modularml.utils.modeling import make_dummy_batch

if TYPE_CHECKING:
    from modularml.core.data_structures.batch import Batch, BatchOutput
    from modularml.core.loss.applied_loss import AppliedLoss
    from modularml.core.optimizer.optimizer import Optimizer


class ModelGraph:
    """
    A directed graph of modular ML components, including FeatureSets and ModelStages.

    Description:
        The `ModelGraph` class manages a collection of `GraphNode` instances, typically
        including `FeatureSet` and `ModelStage` objects. It supports graph construction,
        connection validation, topological sorting, and shared/global optimization logic
        across stages.

        Nodes are connected via their input/output labels, and the graph ensures
        structural consistency (e.g., no cycles, valid dependencies). The graph also
        handles optimizer assignment, either through per-stage optimizers or a shared
        graph-level optimizer (if all stages are compatible).

    Responsibilities:
        - Validate node connectivity and enforce directed acyclic structure
        - Manage graph traversal and stage ordering
        - Optionally assign a shared optimizer to all trainable stages
        - Provide methods to modify, insert, or replace nodes
        - Support staged training across a modular pipeline

    Example Usage:
    ```python
        fs1, model1, model2 = ..., ..., ...
        graph = ModelGraph(nodes=[fs1, model1, model2], optimizer=shared_optimizer)
        graph.build_all()
        graph.train_step(...)
    ```
    """

    def __init__(self, nodes: list[GraphNode], optimizer: Optimizer | None = None):
        """
        Initialize a ModelGraph from a list of modular nodes and an optional global optimizer.

        Args:
            nodes (List[GraphNode]):
                A list of nodes (e.g., `FeatureSet`, `ModelStage`) that form the components
                of the graph. Each node must have a unique `label`.

            optimizer (Optional[Optimizer], optional):
                A shared optimizer to use for all `ModelStage` nodes that require one. If provided,
                the graph will ensure that all such stages have a matching backend and override
                any stage-level optimizers. If not provided, each stage that requires an optimizer
                must define one locally.

        Raises:
            ValueError:
                If duplicate node labels are provided or if graph connectivity is invalid.
            RuntimeError:
                If required optimizers are missing or if backends are incompatible.

        """
        self._nodes: dict[str, GraphNode] = {node.label: copy.deepcopy(node) for node in nodes}

        # Reset node cache
        self._invalidate_node_label_cache()

        # Validate graph connections of provided nodes & sort
        self._validate_graph_connections()
        self._sorted_node_labels = self._topological_sort()

        # If an optimizer is provided, check that:
        # 1. all optimizer-requiring stages have same backend
        # 2. warn if stages have their own optimizer (will be overwritten)
        self._optimizer = optimizer
        self._nodes_req_opt: dict[str, ModelStage] | None = None  # all nodes that require an optimizer
        self._opt_built_from_nodes: set[str] = set()  # nodes used in the current training_phase to build optimizer
        self._validate_optimizer()

        self._built = False

    # ==========================================
    # Properties & Dunders
    # ==========================================
    def _invalidate_node_label_cache(self):
        """
        Invalidate cached lists of source and connected nodes.

        Call this whenever nodes are added/removed or connections change.
        """
        self._source_node_labels: list[str] = []
        self._connected_node_labels: list[str] = []

    def _update_node_label_cache(self):
        """Recompute and cache the lists of source and connected node labels."""
        self._invalidate_node_label_cache()

        for label, node in self._nodes.items():
            if node.allows_upstream_connections:
                self._connected_node_labels.append(label)
            else:
                self._source_node_labels.append(label)

    @property
    def source_node_labels(self) -> list[str]:
        """
        Labels of all nodes that do not have any inputs (e.g., FeatureSets).

        Returns:
            list[str]: Node labels that have no upstream dependencies.

        """
        if self._source_node_labels is None or len(self._source_node_labels) == 0:
            self._update_node_label_cache()
        return self._source_node_labels

    @property
    def connected_node_labels(self) -> list[str]:
        """
        Labels of all nodes that have at least one input (e.g., ModelStages, MergeStages).

        Returns:
            list[str]: Node labels that depend on one or more upstream nodes.

        """
        if self._connected_node_labels is None or len(self._connected_node_labels) == 0:
            self._update_node_label_cache()
        return self._connected_node_labels

    @property
    def is_built(self) -> bool:
        return self._built

    def __repr__(self):
        msg = str(self)
        for node in self._nodes.values():
            msg += f"\n  + {node!s}"
        return msg

    def __str__(self):
        return f"ModelGraph (src_nodes={len(self.source_node_labels)}, conn_nodes={len(self.connected_node_labels)})"

    # ==========================================
    # Error Checking Methods
    # ==========================================
    def _validate_graph_connections(self):
        """
        Validates the internal graph structure.

        Perform the following checks:
        - Ensures nodes are valid GraphNode instances.
        - Propagates inputs to upstream node outputs.
        - Validates input/output limits.
        - Ensures the graph is a DAG (no cycles).
        - Ensures all nodes are reachable from at least one base node.

        Raises:
            TypeError: If any node is not a GraphNode.
            KeyError: If a node references a non-existent input.
            ValueError: If input/output constraints are violated or if a cycle is detected.
            UserWarning: If unreachable nodes are found or mixed backends are used.

        """
        used_backends = []  # record all node backends (for checking)
        frontier = []  # get base nodes (for traversal / connection checks)

        # Ensure node inherits from GraphNode, and input/ouput properties are fully set
        for label, node in self._nodes.items():
            if not isinstance(node, GraphNode):
                msg = f"ModelGraph nodes must be of type GraphNode. Received: {node}"
                raise TypeError(msg)

            # Record backend for later checking
            if isinstance(node, ModelStage):
                used_backends.append(node.backend)

            # Record base nodes (ie, featuresets)
            if not node.allows_upstream_connections:
                frontier.append(label)

            # Ensure that the .outputs properties of all nodes is updated to match the .inputs of all other nodes
            for ups_lbl in node.get_upstream_nodes(
                error_mode=ErrorMode.IGNORE,
            ):  # returns empty list if error occurs
                # Ensure that this input exists in all provided nodes
                if ups_lbl not in self._nodes:
                    msg = f"Upstream node `{ups_lbl}` for node `{label}` not found in ModelGraph."
                    raise KeyError(msg)

                # Add output connection to this input node
                ups_node = self._nodes[ups_lbl]
                if label not in ups_node.get_downstream_nodes(error_mode=ErrorMode.IGNORE):
                    ups_node.add_downstream_node(label)

                # Validate input/output limits
                if (
                    ups_node.max_downstream_nodes is not None
                    and len(ups_node.get_downstream_nodes(error_mode=ErrorMode.IGNORE)) > ups_node.max_downstream_nodes
                ):
                    msg = f"Node '{ups_lbl}' exceeds max_downstream_nodes ({ups_node.max_downstream_nodes})."
                    raise ValueError(msg)

        # Warn if using mixed backend: not thoroughly tested
        if len(set(used_backends)) > 1:
            warnings.warn(
                "Mixed backends detected in ModelGraph. Though allowed, minimal testing has been "
                "conducted. Gradient flow may break during training.",
                category=UserWarning,
                stacklevel=2,
            )

        # Ensure is DAG (check for cycles)
        visited = set()
        visiting = set()

        def dfs(node_label: str):
            """Depth first search."""
            if node_label in visiting:
                msg = f"Cycle detected in graph at node '{node_label}'. Graph must be acyclic."
                raise ValueError(msg)
            if node_label in visited:
                return
            visiting.add(node_label)

            for dwn_lbl in self._nodes[node_label].get_downstream_nodes(
                error_mode=ErrorMode.IGNORE,
            ):
                dfs(dwn_lbl)

            visiting.remove(node_label)
            visited.add(node_label)

        # Perform depth-first-search starting at base nodes (i.e., feature sets)
        for root_label in frontier:
            dfs(root_label)

        # Ensure reachability of all nodes
        reachable = set()
        queue = list(frontier)

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            current_node = self._nodes[current]
            queue.extend(
                current_node.get_downstream_nodes(
                    error_mode=ErrorMode.IGNORE,
                ),
            )

        unreachable = set(self._nodes) - reachable
        if unreachable:
            warnings.warn(
                f"Unreachable nodes detected in ModelGraph: {sorted(unreachable)}.",
                category=UserWarning,
                stacklevel=2,
            )

    def _topological_sort(self) -> list[str]:
        """
        Perform a topological sort of the ModelGraph using Kahn's algorithm.

        Returns:
            List[str]: A list of node labels in topological (execution) order.

        Raises:
            ValueError: If a cycle is detected in the graph.

        """
        in_degree = defaultdict(int)  # Number of incoming edges (keyed by node label)
        children = defaultdict(list)  # Outgoing edges (keyed by node label)
        all_node_names = set(self._nodes.keys())

        # Initialize in-degrees
        for label in all_node_names:
            in_degree[label] = 0

        # Record in-degree (number of inputs) and out-degree for each node
        for label, node in self._nodes.items():
            # Get parents of this node
            parents: list[str] = node.get_upstream_nodes(error_mode=ErrorMode.IGNORE)
            for parent in parents:
                if parent not in self._nodes:
                    msg = f"Invalid upstream_node `{parent}` for node `{label}`."
                    raise KeyError(msg)
                in_degree[label] += 1
                children[parent].append(label)

        # Init a queue with base nodes (no inputs)
        sorted_node_labels: list[str] = []
        queue = deque([label for label in all_node_names if in_degree[label] == 0])
        while queue:
            current = queue.popleft()
            sorted_node_labels.append(current)
            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_node_labels) != len(all_node_names):
            unresolved = all_node_names - set(sorted_node_labels)
            msg = f"Cyclic dependency detected in ModelGraph: {unresolved}"
            raise ValueError(msg)

        return sorted_node_labels

    def _validate_optimizer(self):
        """
        Validate and assign optimizers to all trainable stages in the graph.

        Description:
            This method ensures that all `ModelStage` nodes which require an optimizer
            (based on their backend) are properly configured with one.

            - If a global optimizer is provided to the `ModelGraph`, it will be assigned
            to all relevant stages. If those stages already define a local optimizer,
            it will be overwritten with a warning.

            - If no global optimizer is provided, then every stage that requires an optimizer
            must have its own stage-level optimizer defined.

            - It also verifies that all optimizers share a consistent backend (e.g., PyTorch).

        Raises:
            RuntimeError: If any stage that requires an optimizer is missing one and no
                        global optimizer is provided.
            RuntimeError: If a global optimizer is provided but its backend doesn't match
                        a stage's backend.
            UserWarning: If a stage has its own optimizer but is being overwritten by the
                        graph-level optimizer.

        """
        # Get nodes that require optimizer (only ModelStages)
        self._nodes_req_opt = {
            label: node
            for label, node in self._nodes.items()
            if isinstance(node, ModelStage) and backend_requires_optimizer(node.backend)
        }

        # Ensure all stages have their own optimizer if global one not provided
        if self._optimizer is None:
            for label, stage in self._nodes_req_opt.items():
                if stage._optimizer is None:
                    msg = (
                        f"ModelStage (`{label}`) is missing an optimizer. "
                        f"Provide one at the stage level or to ModelGraph."
                    )
                    raise RuntimeError(msg)

        # Ensure all stages have the same backend
        else:
            used_backends = []
            for label, node in self._nodes_req_opt.items():
                used_backends.append(node.backend)
                # Overwrite existing optimizers at stage-level (and warn)
                if node._optimizer is not None:
                    warnings.warn(
                        (
                            f"Optimizer were provided to ModelGraph and an underlying ModelStage (`{label}`). "
                            f"The stage-level optimizer will be overwritten."
                        ),
                        category=UserWarning,
                        stacklevel=2,
                    )
                    node._optimizer = None

            # Warn if using mixed backend: not thoroughly tested
            if len(set(used_backends)) > 1:
                msg = (
                    "A global optimizer was provided to ModelGraph, but the underlying stages have "
                    "differing backends. All backends must match to use a single optimizer."
                )
                raise RuntimeError(msg)
            self._optimizer.backend = used_backends[0]

    # ==========================================
    # ModelGraph Node Modifiers
    # ==========================================
    def copy(self) -> ModelGraph:
        """
        Create a deep copy of the ModelGraph.

        This duplicates all GraphNode instances (FeatureSets, ModelStages, etc.) and the optional global optimizer.
        Connections and internal structure are preserved, and a new ModelGraph is returned with its own rebuilt graph.

        Returns:
            ModelGraph: A completely independent copy of the current graph.

        """
        # Deep copy all nodes (ModelStages + FeatureSets)
        copied_nodes = [copy.deepcopy(node) for node in self._nodes.values()]

        # Deep copy optimizer if it exists
        copied_optimizer = copy.deepcopy(self._optimizer) if self._optimizer is not None else None

        # Create new ModelGraph from copied components
        new_graph = ModelGraph(nodes=copied_nodes, optimizer=copied_optimizer)

        return new_graph

    def add(self, node: GraphNode, *, inplace: bool = True) -> ModelGraph | None:
        """
        Add a new node to the graph.

        This adds the node to the graph and rebuilds all connections. If `inplace` is False,
        a copy of the graph is created with the node added there.

        Args:
            node (GraphNode): Node to add (must have a unique label).
            inplace (bool): Whether to modify the current graph or return a new copy.

        Returns:
            Optional[ModelGraph]: The updated graph if `inplace=False`, otherwise None.

        Raises:
            TypeError: If `node` is not a GraphNode.
            ValueError: If a node with the same label already exists.

        """
        if not inplace:
            new_graph = self.copy()
            new_graph.add(node, inplace=True)
            return new_graph

        if not isinstance(node, GraphNode):
            msg = f"Node must inherit from GraphNode. Received: {node}"
            raise TypeError(msg)

        if node.label in self._nodes:
            msg = f"Node already exists with label: `{node.label}`. Use `replace` to replace it with a new node."
            raise ValueError(msg)

        self._nodes[node.label] = node
        self._update_node_label_cache()
        self.build_all(reset=True)
        return None

    def replace(self, node: GraphNode, *, inplace: bool = True) -> ModelGraph | None:
        """
        Replace an existing node in the graph by label.

        Keeps all connections to/from the replaced node intact.
        If `inplace` is False, a modified copy is returned instead.

        Args:
            node (GraphNode): Node to replace the existing node with the same label.
            inplace (bool): Whether to modify the current graph or return a new copy.

        Returns:
            Optional[ModelGraph]: The updated graph if `inplace=False`, otherwise None.

        Raises:
            TypeError: If `node` is not a GraphNode.
            ValueError: If no existing node matches the label of `node`.

        """
        if not inplace:
            new_graph = self.copy()
            new_graph.replace(node, inplace=True)
            return new_graph

        if not isinstance(node, GraphNode):
            msg = f"Node must inherit from GraphNode. Received: {node}"
            raise TypeError(msg)

        if node.label not in self._nodes:
            msg = f"There are no existing nodes to replace with label: `{node.label}`. Use `add` to add it instead."
            raise ValueError(msg)

        self._nodes[node.label] = node
        self._update_node_label_cache()
        self.build_all(reset=True)
        return None

    def insert(
        self,
        node: GraphNode,
        before: str | None = None,
        after: str | None = None,
        *,
        inplace: bool = True,
    ) -> ModelGraph | None:
        """
        Insert a new node between existing nodes.

        The graph is rewired such that:
        - If both `after` and `before` are provided, the node is inserted between them (replacing any existing connection).
        - If only `before` is provided, all previous inputs to `before` are redirected through the new node.
        - If only `after` is provided, all outputs from `after` are redirected through the new node.

        Args:
            node (GraphNode): The new node to insert.
            before (Optional[str]): Label of a downstream node.
            after (Optional[str]): Label of an upstream node.
            inplace (bool): If False, returns a copy with the insertion applied.

        Returns:
            Optional[ModelGraph]: The updated graph if `inplace=False`, otherwise None.

        Raises:
            TypeError: If `node` is not a GraphNode.
            ValueError: If `before` or `after` are specified but not found in the graph.

        """
        # To insert between two specific node, before & after must be specified
        # To insert between a node and all prior outputs, only after should be used
        # To insert a node before a node, only before should be used
        if not isinstance(node, GraphNode):
            msg = f"Node must be of type FeatureSet or ModelStage. Received: {node}"
            raise TypeError(msg)
        if after is not None and after not in self._nodes:
            msg = f"No existing node with label `{after}`"
            raise ValueError(msg)
        if before is not None and before not in self._nodes:
            msg = f"No existing node with label `{before}`"
            raise ValueError(msg)

        if not inplace:
            new_graph = self.copy()
            new_graph.insert(
                node=node,
                after=after,
                before=before,
                inplace=True,
            )
            return new_graph

        if before is not None and after is not None:
            # if a connection between before and after exists, this should replace it
            # otherwise, insert a new connection
            # after -> before   >>>   after -> new_node -> before
            downstream_node = self._nodes[before]
            upstream_node = self._nodes[after]

            # Ensure new node defines the proper connections
            node.add_upstream_node(after, error_mode=ErrorMode.COERCE)  # ignore if already exists
            node.add_downstream_node(before, error_mode=ErrorMode.COERCE)  # ignore if already exists

            # Update downstream_node to input from new node (remove any connection to upstream_node)
            downstream_node.remove_upstream_node(
                upstream_node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore it connection doesn't exist
            downstream_node.add_upstream_node(
                node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore if already exists

            # Update upstream_node to output to new node (remove any connection to downstream_node)
            upstream_node.remove_downstream_node(
                downstream_node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore it connection doesn't exist
            upstream_node.add_downstream_node(
                node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore if already exists

        # insert before
        elif before is not None:
            # moves all input connections on the downstream_node (`before`) onto new_node
            # downstream_node inputs only from new_node

            # Move downstream_node (`before`) inputs to new_node inputs
            downstream_node = self._nodes[before]
            node.set_upstream_nodes(
                upstream_nodes=downstream_node.get_upstream_nodes(
                    error_mode=ErrorMode.COERCE,
                ),
                error_mode=ErrorMode.COERCE,
            )

            # Update downstream_node to input only from new_node
            downstream_node.clear_upstream_nodes()
            downstream_node.add_upstream_node(node.label)

            # Update new_node to only output to downstream_node
            node.clear_downstream_nodes()
            node.add_downstream_node(
                downstream_node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore if already exists

            # Updates all upstream nodes that used to output to downstream_node to now output to new_node
            x = self._sorted_node_labels.index(
                before,
            )  # end at downstream_node to save time
            for lbl in self._sorted_node_labels[:x]:
                n = self._nodes[lbl]
                if n.allows_downstream_connections and downstream_node.label in n.get_downstream_nodes():
                    n.remove_downstream_node(downstream_node.label)  # remove downstream node
                    n.add_downstream_node(node.label)  # replace with new node

        # insert after
        elif after is not None:
            # moves all output connections on the upstream_node (`after`) onto new_node
            # new_node inputs only from upstream_node

            # Move upstream_node (`after`) outputs to new_node outputs
            upstream_node = self._nodes[after]
            node.set_downstream_nodes(
                downstream_nodes=upstream_node.get_downstream_nodes(error_mode=ErrorMode.COERCE),
                error_mode=ErrorMode.COERCE,
            )

            # Update upstream_node to output only to new_node
            upstream_node.clear_downstream_nodes()
            upstream_node.add_downstream_node(node.label)

            # Update new_node to only input from upstream node
            node.clear_upstream_nodes()
            node.add_upstream_node(
                upstream_node.label,
                error_mode=ErrorMode.COERCE,
            )  # ignore if already exists

            # Updates all downstream nodes that used to input from upstream_node to now input from new_node
            x = self._sorted_node_labels.index(
                after,
            )  # start from upstream node to save time
            for lbl in self._sorted_node_labels[x:]:
                n = self._nodes[lbl]
                if n.allows_upstream_connections and upstream_node.label in n.get_upstream_nodes():
                    n.remove_upstream_node(upstream_node.label)  # remove upstream node
                    n.add_upstream_node(node.label)  # replace with new node

        # Add new node
        self.add(node=node, inplace=True)
        return None

    def remove(self, node: GraphNode | str, *, inplace: bool = True):
        """
        Remove a node from the graph by label or instance.

        All connections to/from the node are removed across the graph.
        If `inplace` is False, a new graph with the node removed is returned.

        Args:
            node (Union[GraphNode, str]): The node or label of the node to remove.
            inplace (bool): Whether to modify the current graph or return a copy.

        Returns:
            Optional[ModelGraph]: The updated graph if `inplace=False`, otherwise None.

        Raises:
            ValueError: If node is not found or of incorrect type.

        """
        if not inplace:
            new_graph = self.copy()
            new_graph.remove(
                node=node,
                inplace=True,
            )
            return new_graph

        if isinstance(node, str):
            if node not in self._nodes:
                msg = f"No node exists with label `{node}`"
                raise ValueError(msg)
            node = self._nodes[node]
        if not isinstance(node, GraphNode):
            msg = f"Node to remove must be a string or GraphNode. Received: {node}"
            raise TypeError(msg)

        # If node has an input (upstream_node), we need to re-assign upstream_node to route to downstream_node
        upstream_connections = node.get_upstream_nodes(error_mode=ErrorMode.COERCE)
        downstream_connections = node.get_downstream_nodes(error_mode=ErrorMode.COERCE)

        # Remove upstream references to "node"
        for upstream_lbl in upstream_connections:
            self._nodes[upstream_lbl].remove_downstream_node(node=node.label)
            # Add new connectiosn to all nodes downstream of "node"
            for downstream_lbl in downstream_connections:
                self._nodes[upstream_lbl].add_downstream_node(downstream_lbl, error_mode=ErrorMode.COERCE)

        # Remove downstream references to "node"
        for downstream_lbl in downstream_connections:
            self._nodes[downstream_lbl].remove_upstream_node(node=node.label)
            # Add new connectiosn to all nodes downstream of "node"
            for upstream_lbl in upstream_connections:
                self._nodes[downstream_lbl].add_upstream_node(upstream_lbl, error_mode=ErrorMode.COERCE)

        # Remove node from graph
        self._nodes.pop(node.label)

        # Update connections and rebuild all models
        self._update_node_label_cache()
        self.build_all(reset=True)
        return None

    # ==========================================
    # Graph Construction Helpers
    # ==========================================
    def _get_input_shapes(self, node: ComputationNode) -> list[ShapeSpec]:
        """Gets all inputs shapes for this node."""
        input_shapes = []
        for inp in node.get_upstream_nodes():
            out_shape = self._nodes[inp].output_shape_spec
            if out_shape is None:
                msg = (
                    f"Cannot infer input for node `{node.label}` because the output_shape is "
                    f"None for upstream node `{self._nodes[inp]}`."
                )
                raise ValueError(msg)
            input_shapes.append(out_shape)
        return input_shapes

    def _build_optimizer(self, nodes_to_build_optimizer_with: list[str]):
        # Build global optimizer if defined
        if self._optimizer is not None:
            if self._optimizer.backend == Backend.TORCH:
                # Collect model parameters from nodes to build optimizer with
                all_model_params = []
                for node_lbl in nodes_to_build_optimizer_with:
                    all_model_params.extend(list(self._nodes[node_lbl]._model.parameters()))

                # Build optimizer will all parameters
                self._optimizer.build(force_rebuild=True, parameters=all_model_params)

            elif self._optimizer.backend == Backend.TENSORFLOW:
                self._optimizer.build(force_rebuild=False)

            elif self._optimizer.backend == Backend.SCIKIT:
                # Scikit-learn optimizers are typically fit internally
                pass

            else:
                raise BackendNotSupportedError(
                    backend=self._optimizer.backend,
                    message="Unknown backend for optimizer building",
                )

        # Update record of nodes used to construct this optimizer
        self._opt_built_from_nodes = set(nodes_to_build_optimizer_with)

    def build_all(self, *, reset: bool = False) -> None:
        """
        Build all ModelStages contained in this ModelGraph.

        This method constructs all computation nodes (e.g., ModelStages) by:
        - Performing shape inference based on dummy inputs from each FeatureSet.
        - Building each stage in topological order.
        - Propagating dummy data forward through the graph to infer input/output shapes.
        - Optionally rebuilding and validating the entire graph structure if `reset=True`.

        If a global optimizer is attached, it is also built using the accumulated
        parameters from all trainable ModelStages.

        Args:
            reset (bool, optional): If True, the graph will be revalidated and
                rebuilt from scratch. This re-computes:
                - Node sorting
                - Optimizer requirements
                - Internal caches
                Defaults to False.

        Raises:
            BackendNotSupportedError: If the optimizer backend is unrecognized.
            ValueError: If shape inference fails for any stage.

        """
        if reset:
            # Validate graph connections of provided nodes & sort
            self._invalidate_node_label_cache()
            self._validate_graph_connections()
            self._sorted_node_labels = self._topological_sort()
            self._nodes_req_opt: dict[str, ModelStage] | None = None
            self._validate_optimizer()

            self._built = False

        # To provide potential output_shape inference of ModelStage leaf nodes,
        # we can perform a forward pass through each stage and record outputs
        # the outputs will propagate the expected target_shapes
        dummy_input_data: dict[str, Batch] = {}
        for lbl in self.source_node_labels:
            node = self._nodes[lbl]
            if isinstance(node, FeatureSet):
                batch = make_dummy_batch(
                    feature_shape=node.feature_shape_spec,
                    target_shape=node.target_shape_spec,
                    batch_size=8,
                )
                dummy_input_data[lbl] = batch

        # Ensure all nodes are built
        cache: dict[str, Batch | BatchOutput] = {}
        for lbl in self._sorted_node_labels:
            node: GraphNode = self._nodes[lbl]

            # If a source node (ie, FeatureSet), just record input data
            if lbl in self.source_node_labels:
                cache[lbl] = dummy_input_data[lbl]
                continue

            # Build node with shape inferrence
            if isinstance(node, ComputationNode) and (not node.is_built or reset):
                # Get all inputs feeding into this node
                input_shapes: list[ShapeSpec] = self._get_input_shapes(node)

                # ModelStage support output shape inferrence
                if isinstance(node, ModelStage):
                    try:
                        output_shapes: list[ShapeSpec] = node.infer_output_shape_spec(input_shapes)
                    except ValueError:
                        output_shapes = None
                    # If a leaf ModelStage and failed to infer output_shapes,
                    # force output_shapes to be the accumulated FeatureSet.target_shape
                    if output_shapes is None and isinstance(node, ModelStage):
                        output_shapes = [
                            ShapeSpec(shapes={"_output_": cache[node.upstream_node].target_shape}),
                        ]

                    # Build node
                    node.build(
                        input_shapes=input_shapes,
                        output_shapes=output_shapes,
                        force=reset,
                    )
                    msg = (
                        f"Built node `{lbl}` with shapes: {node.input_shape_spec.merged_shape}"
                        f" -> {node.output_shape_spec.merged_shape}"
                    )
                    print(msg)

                    # Cache outputs of this newly built node
                    output: BatchOutput = node.forward(node.get_input_batch(cache))
                    cache[lbl] = output

                # MergeStage only accepts an input_shape, but also requires a backend
                elif isinstance(node, MergeStage):
                    backend = Backend.SCIKIT if self._optimizer is None else self._optimizer.backend
                    node.build(input_shapes=input_shapes, backend=backend)
                    msg = (
                        f"Built node `{lbl}` with shapes: {list(node.input_shape_spec.values())}"
                        f" -> {node.merged_shape.merged_shape}"
                    )
                    print(msg)

                    # Cache outputs of this newly built node
                    output: list[BatchOutput] = node.forward(node.get_input_batch(cache))
                    cache[lbl] = output

                else:
                    msg = f"Expected ModelStage or MergeStage. Received: {type(node)}"
                    raise TypeError(msg)

        # Build optimizer with all optimizable nodes
        self._build_optimizer(self._nodes_req_opt)

        self._built = True

    def get_node_source(self, node: GraphNode | str) -> str | tuple[str] | None:
        """
        Recursively trace the ultimate source FeatureSet node(s) that feed into the given node.

        A "source node" in this context means one of the graph's root FeatureSets
        (i.e., labels in `self._source_node_labels`) that provides original data
        with no upstream dependencies.

        Args:
            node (GraphNode | str):
                The target node for which to trace back sources. Can be either a
                GraphNode instance or its string label.

        Returns:
            str | tuple[str] | None:
                - If exactly one source node is found, returns its label as a string.
                - If multiple distinct source nodes are found, returns a tuple of their labels.
                - If no sources are found (e.g., orphaned node), returns None.

        Notes:
            - Traverses all upstream connections recursively, not just direct parents.
            - Useful for determining lineage of a ModelStage output
            (e.g., which FeatureSet(s) provided its training/evaluation data).
            - If the node is itself a source FeatureSet, it will be returned as the sole source.

        """
        node_instance = node
        if isinstance(node, str):
            node_instance = self._nodes[node]

        visited: set[str] = set()

        def _collect_sources(n: GraphNode) -> set[str]:
            if n.label in visited:
                return set()  # prevent cycles
            visited.add(n.label)

            # If this node is itself a source, stop here
            if n.label in self._source_node_labels:
                return {n.label}

            # Otherwise, recurse upstream
            sources = set()
            for upstream in n.get_upstream_nodes(error_mode=ErrorMode.COERCE):
                sources |= _collect_sources(self._nodes[upstream])
            return sources

        sources = _collect_sources(node_instance)

        if not sources:
            return None
        if len(sources) == 1:
            return next(iter(sources))
        return tuple(sources)

    # ==========================================
    # Forward Pass / Direct Calls
    # ==========================================
    def dummy_foward(self, batch_size: int = 8):
        """
        Run a forward pass through the entire ModelGraph using dummy inputs.

        This method constructs fake `Batch` inputs for each `FeatureSet` using the
        registered `feature_shape`, and forwards them through all downstream nodes.

        This is useful for:
        - Verifying connectivity and shape compatibility between nodes
        - Auto-initializing unbuilt stages
        - Debugging pipelines without real data

        Args:
            batch_size (int, optional): Number of samples to generate per input FeatureSet. Defaults to 8.

        Returns:
            Dict[str, Batch | BatchOutput]: A dictionary of node outputs keyed by node label.

        """
        # Generate source data (ie, generate a batch from each FeatureSet)
        batches: dict[str, Batch] = {}
        for lbl in self.source_node_labels:
            node = self._nodes[lbl]
            if isinstance(node, FeatureSet):
                batch = make_dummy_batch(
                    feature_shape=node.feature_shape_spec,
                    batch_size=batch_size,
                )
                batches[lbl] = batch

        res = self.forward(batches)
        return res

    def forward(self, batches: dict[str, Batch]) -> dict[str, Batch | BatchOutput]:
        """
        Perform a full forward pass through the graph with real input data.

        All source nodes (FeatureSets) must be provided as input. The graph
        will compute outputs for each downstream node based on connectivity.

        Args:
            batches (dict[str, Batch]):
                Dictionary of input data, keyed by FeatureSet label.

        Returns:
            dict[str, Batch | BatchOutput]:
                Dictionary of outputs from each node, keyed by node label.
                FeatureSets will return their original `Batch`; downstream stages
                return `BatchOutput`.

        Raises:
            ValueError: If any required FeatureSet input is missing.
            TypeError: If an unknown node type is encountered.

        """
        missing_inputs = []
        for lbl in self.source_node_labels:
            if lbl not in batches:
                missing_inputs.append(lbl)
        if missing_inputs:
            msg = f"The batches provided to ModelGraph is missing data from required inputs. Missing: {missing_inputs}"
            raise ValueError(msg)

        # Stores output from each node in ModelGraph
        cache: dict[str, Batch | BatchOutput] = {}
        for lbl in self._sorted_node_labels:
            node = self._nodes[lbl]

            # If a source node (ie, FeatureSet), just record input data
            if lbl in self.source_node_labels:
                cache[lbl] = batches[lbl]
                continue

            # Otherwise, perform forward pass of ComputationNodes
            if isinstance(node, ComputationNode):
                for ups_node in node.get_upstream_nodes():
                    if ups_node not in cache:
                        msg = f"Missing upstream node `{ups_node}` for node `{lbl}`"
                        raise ValueError(msg)
                output: BatchOutput = node.forward(node.get_input_batch(cache))
                cache[lbl] = output

            else:
                msg = f"Unknown node type encountered during forward pass: {node}"
                raise TypeError(msg)

        return cache

    # ==========================================
    # Train & Eval Methods
    # ==========================================
    def _stagewise_train_step(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
        nodes_to_iterate_over: list[str],
    ) -> StepResult:
        """
        Execute training by calling `train_step` on each stage individually.

        Only used when the ModelGraph does not have a global optimizer. Each
        `ModelStage` handles its own optimization independently.

        Args:
            batch_input (dict[str, Batch]): Input data keyed by FeatureSet label.
            losses (dict[str, list[AppliedLoss]]): Applied losses keyed by node label.
            nodes_to_iterate_over (list[str]): A list of nodes defining the minimum subgraph needing to be trained.

        Returns:
            StepResult: Aggregated losses and outputs from the training step.

        Raises:
            ValueError: If a trainable node is missing loss definitions.
            TypeError: If any node is not a ComputationNode.

        """
        # Cache all stage outputs
        cache: dict[str, Batch | BatchOutput] = dict(batch_input)
        loss_cache: LossCollection = None

        # Forward pass + collect outputs
        for lbl in nodes_to_iterate_over:
            if lbl in self.source_node_labels:
                continue

            node = self._nodes[lbl]
            node_losses = losses.get(lbl)
            if not isinstance(node, ComputationNode):
                msg = f"Training can only be performed on ComputationNodes. Received: {node}."
                raise TypeError(msg)

            # Use train_step if trainable
            if isinstance(node, TrainableMixin) and not node.freeze:
                if node_losses is None:
                    msg = f"Node `{lbl}` is set to train but has no applied losses."
                    raise ValueError(msg)
                step_res = node.train_step(batch_input=cache, losses=node_losses)

            # Use eval_step if not trainable but is evaluable
            elif isinstance(node, EvaluableMixin):
                if node_losses:
                    warnings.warn(
                        f"Node `{lbl}` has losses applied during train_step, but it is frozen. "
                        f"These losses will not contribute to optimizer stepping.",
                        stacklevel=2,
                    )
                step_res = node.eval_step(batch_input=cache, losses=node_losses)

            # Use manual forward + loss comp otherwise (no optimizer contribution)
            else:
                model_output: BatchOutput = node.forward(node.get_input_batch(cache))
                loss_records: list[LossRecord] = []
                if node_losses is not None:
                    for loss in node_losses:
                        weighted_raw_loss = loss.compute(
                            batch_input=cache,
                            model_outputs={lbl: model_output},
                        )
                        lr = LossRecord(
                            value=weighted_raw_loss,
                            label=loss.label,
                            contributes_to_update=False,  # does not contribute to loss stepping
                        )
                        loss_records.append(lr)
                lc = LossCollection(records=loss_records)
                step_res = StepResult(losses=lc, node_outputs={lbl: model_output})

            # Cache stage outputs
            cache[lbl] = step_res.node_outputs[lbl]

            # Merge losses into a single LossCollection
            if loss_cache is None:
                loss_cache = step_res.losses
            else:
                loss_cache += step_res.losses

        return StepResult(
            losses=loss_cache,
            node_outputs={k: v for k, v in cache.items() if k in self.connected_node_labels},
        )

    def _train_step_torch(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
        nodes_to_iterate_over: list[str],
    ) -> StepResult:
        """
        Perform full-graph optimization using a PyTorch-based optimizer.

        This method performs a single forward and backward pass across all
        trainable stages, then steps the optimizer.

        Args:
            batch_input (dict[str, Batch]): Input batches from FeatureSets.
            losses (dict[str, list[AppliedLoss]]): Loss functions to apply per stage.
            nodes_to_iterate_over (list[str]): A list of nodes defining the minimum subgraph needing to be trained.

        Returns:
            StepResult: Aggregated loss values, stage outputs, and gradients.

        Raises:
            RuntimeError: If no trainable loss was found.
            TypeError: If a non-ComputationalNode is encountered.

        """
        # Cache all stage outputs
        cache: dict[str, Batch | BatchOutput] = dict(batch_input)
        loss_records: list[LossRecord] = []

        # Set optimizer
        self._optimizer.zero_grad()

        # Forward pass + collect outputs
        for lbl in nodes_to_iterate_over:
            if lbl in self.source_node_labels:
                continue

            node = self._nodes[lbl]
            node_losses = losses.get(lbl)
            if not isinstance(node, ComputationNode):
                msg = f"Training can only be performed on ComputationNodes. Received: {node}."
                raise TypeError(msg)

            if isinstance(node, TrainableMixin) and not node.freeze:
                # Perform manual train + loss comp (to ensure no autograd breakage)
                if hasattr(node.model, "train"):
                    node.model.train()
                model_output: BatchOutput = node.forward(node.get_input_batch(cache))

                # Compute stage loss (store entire LossResult for later computation)
                if node_losses is not None:
                    for loss in node_losses:
                        weighted_raw_loss = loss.compute(
                            batch_input=cache,
                            model_outputs={lbl: model_output},
                        )
                        lr = LossRecord(
                            value=weighted_raw_loss,
                            label=loss.label,
                            contributes_to_update=True,
                        )
                        loss_records.append(lr)

                cache[lbl] = model_output
                continue

            if node_losses:
                warnings.warn(
                    f"Node `{lbl}` has losses applied during train_step, but it is not a "
                    "subclass of TrainableMixin. These losses will not contribute to "
                    "optimizer stepping.",
                    stacklevel=2,
                )

            if hasattr(node.model, "eval"):
                node.model.eval()
            model_output: BatchOutput = node.forward(node.get_input_batch(cache))

            # Compute stage loss (store entire LossResult for later computation)
            if node_losses is not None:
                for loss in node_losses:
                    weighted_raw_loss = loss.compute(
                        batch_input=cache,
                        model_outputs={lbl: model_output},
                    )
                    lr = LossRecord(
                        value=weighted_raw_loss,
                        label=loss.label,
                        contributes_to_update=False,
                    )
                    loss_records.append(lr)

            cache[lbl] = model_output
            continue

        # Perform optimization stepping
        if not loss_records:
            raise RuntimeError("Optimizer stepping cannot be performed because no recorded losses.")

        lc = LossCollection(records=loss_records)
        lc.trainable.backward()
        self._optimizer.step()

        return StepResult(
            losses=lc,
            node_outputs={k: v for k, v in cache.items() if k in self.connected_node_labels},
        )

    def _train_step_tensorflow(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
        nodes_to_iterate_over: list[str],
    ) -> StepResult:
        """
        Perform full-graph optimization using a TensorFlow-based optimizer.

        Executes forward pass within a `tf.GradientTape` and applies gradients
        to all trainable variables.

        Args:
            batch_input (dict[str, Batch]): Input batches keyed by FeatureSet.
            losses (dict[str, list[AppliedLoss]]): Loss functions to compute.
            nodes_to_iterate_over (list[str]): A list of nodes defining the minimum subgraph needing to be trained.

        Returns:
            StepResult: Object containing loss values and model outputs.

        Raises:
            RuntimeError: If no optimizable loss was found.
            TypeError: If a non-ComputationalNode is encountered.

        """
        # Cache all stage outputs
        cache: dict[str, Batch | BatchOutput] = dict(batch_input)
        loss_records: list[LossRecord] = []

        # Set optimizer
        self._optimizer.zero_grad()

        # Collect trainable variables for optimization
        trainable_vars = []

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass + collect outputs
            for lbl in nodes_to_iterate_over:
                if lbl in self.source_node_labels:
                    continue

                node = self._nodes[lbl]
                node_losses = losses.get(lbl)
                if not isinstance(node, ComputationNode):
                    msg = f"Training can only be performed on ComputationNodes. Received: {node}."
                    raise TypeError(msg)

                if isinstance(node, TrainableMixin) and not node.freeze:
                    # Collect trainable model variables
                    if hasattr(node.model, "trainable_variables"):
                        trainable_vars += list(node.model.trainable_variables)

                    model_output: BatchOutput = node.forward(
                        node.get_input_batch(cache),
                        training=True,
                    )

                    # Compute stage loss (store entire LossResult for later computation)
                    if node_losses is not None:
                        for loss in node_losses:
                            weighted_raw_loss = loss.compute(
                                batch_input=cache,
                                model_outputs={lbl: model_output},
                            )
                            lr = LossRecord(
                                value=weighted_raw_loss,
                                label=loss.label,
                                contributes_to_update=True,
                            )
                            loss_records.append(lr)

                    cache[lbl] = model_output
                    continue

                if node_losses:
                    warnings.warn(
                        f"Node `{lbl}` has losses applied during train_step, but it is not a "
                        "subclass of TrainableMixin. These losses will not contribute to "
                        "optimizer stepping.",
                        stacklevel=2,
                    )
                model_output: BatchOutput = node.forward(
                    node.get_input_batch(cache),
                    training=False,
                )

                # Compute stage loss (store entire LossResult for later computation)
                if node_losses is not None:
                    for loss in node_losses:
                        weighted_raw_loss = loss.compute(
                            batch_input=cache,
                            model_outputs={lbl: model_output},
                        )
                        lr = LossRecord(
                            value=weighted_raw_loss,
                            label=loss.label,
                            contributes_to_update=False,
                        )
                        loss_records.append(lr)

                cache[lbl] = model_output
                continue

        # Perform optimization stepping
        if not loss_records:
            raise RuntimeError("Optimizer stepping cannot be performed because no recorded losses.")

        lc = LossCollection(records=loss_records)
        grads = tape.gradient(lc.trainable, trainable_vars)
        if any(g is None for g in grads):
            raise RuntimeError("Some gradients are None. Check your graph and loss construction.")
        self._optimizer.step(grads=grads, variables=trainable_vars)
        del tape  # Clean up persistent tape

        return StepResult(
            losses=lc,
            node_outputs={k: v for k, v in cache.items() if k in self.connected_node_labels},
        )

    def _graphwise_train_step(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
        nodes_to_iterate_over: list[str],
    ) -> StepResult:
        """
        Perform a graph-level optimization step.

        This method delegates to a backend-specific implementation
        (PyTorch or TensorFlow) and performs a full optimization step
        across all stages simultaneously.

        Args:
            batch_input (dict[str, Batch]): Input data for the graph.
            losses (dict[str, list[AppliedLoss]]): Loss functions to apply.
            nodes_to_iterate_over (list[str]): A list of nodes defining the minimum subgraph needing to be trained.

        Returns:
            StepResult: Contains total loss, individual losses, and outputs.

        Raises:
            ValueError: If the optimizer backend is unsupported.

        """
        if self._optimizer.backend == Backend.TORCH:
            return self._train_step_torch(
                batch_input=batch_input,
                losses=losses,
                nodes_to_iterate_over=nodes_to_iterate_over,
            )

        if self._optimizer.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(
                batch_input=batch_input,
                losses=losses,
                nodes_to_iterate_over=nodes_to_iterate_over,
            )

        msg = f"Unknown backend for optimization: {self._optimizer.backend}"
        raise ValueError(msg)

    def _rebuild_optimizer_if_needed(self, trainable_stages: list[str]):
        """Rebuilds the graph-level optimizer only if the set of trainable stages has changed."""
        stage_set = set(trainable_stages)

        # Only rebuild if the trainable set has changed
        if stage_set == self._opt_built_from_nodes:
            return
        msg = (
            "The set of trainable nodes in ModelGraph have changed. "
            f"The optimizer is being rebuilt for nodes: {stage_set}"
        )
        warnings.warn(
            message=msg,
            category=UserWarning,
            stacklevel=2,
        )

        # Build optimizer with optimizable nodes in this training phase
        self._build_optimizer(stage_set)

    def train_step(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
        trainable_stages: list[str],
    ) -> StepResult:
        """
        Train the graph using either stagewise or global optimization.

        Freezes all stages not listed in `trainable_stages`.

        If a global optimizer is set, this performs joint optimization across
        all trainable nodes. Otherwise, each trainable stage optimizes independently.

        Args:
            batch_input (dict[str, Batch]):
                Dictionary of FeatureSet input data.

            losses (dict[str, list[AppliedLoss]]):
                Loss functions to apply to each stage.

            trainable_stages (list[str]):
                List of ModelStage labels to train. Others will be frozen.

        Returns:
            StepResult: Aggregated loss and output result of training step.

        """
        # Freeze/unfreeze stages
        for node_lbl, node in self._nodes.items():
            if isinstance(node, TrainableMixin):
                node.freeze = node_lbl not in trainable_stages

        # Determine which nodes actually need to be iterated over
        nodes_to_iterate_over: set[str] = set()
        # region: For each trainable stage, add all its upstream nodes (and itself) to the list
        for stage in trainable_stages:
            if stage not in self._sorted_node_labels:
                msg = f"Trainable stage '{stage}' not found in graph."
                raise ValueError(msg)

            idx = self._sorted_node_labels.index(stage)
            # Add all nodes up to and including this one
            nodes_to_iterate_over.update(self._sorted_node_labels[: idx + 1])
        # Preserve topological order by filtering self._sorted_node_labels
        nodes_to_iterate_over = [lbl for lbl in self._sorted_node_labels if lbl in nodes_to_iterate_over]
        # endregion

        # Case 1: Graph-level training
        if self._optimizer is not None:
            # Rebuild global optimizer (only if set of trainable stages changed)
            self._rebuild_optimizer_if_needed(trainable_stages=trainable_stages)
            # Run graphwise training logic
            return self._graphwise_train_step(
                batch_input=batch_input,
                losses=losses,
                nodes_to_iterate_over=nodes_to_iterate_over,
            )

        # Case 2: Stage-level training
        return self._stagewise_train_step(
            batch_input=batch_input,
            losses=losses,
            nodes_to_iterate_over=nodes_to_iterate_over,
        )

    def eval_step(
        self,
        batch_input: dict[str, Batch],
        losses: dict[str, list[AppliedLoss]],
    ) -> StepResult:
        """
        Evaluate all stages in the graph without updating parameters.

        All stages are frozen before evaluation.

        Args:
            batch_input (dict[str, Batch]): Input data keyed by FeatureSet.
            losses (dict[str, list[AppliedLoss]]): Losses to apply per stage.

        Returns:
            StepResult: Loss values and stage outputs for the entire graph.

        Raises:
            TypeError: If a stage is not evaluable.

        """
        # Freeze all stages
        for node in self._nodes.values():
            if isinstance(node, TrainableMixin):
                node.freeze = True

        # Cache all stage outputs
        cache: dict[str, Batch | BatchOutput] = dict(batch_input)
        loss_cache: LossCollection = None

        # Forward pass + collect outputs
        for lbl in self._sorted_node_labels:
            if lbl in self.source_node_labels:
                continue

            node = self._nodes[lbl]
            node_losses = losses.get(lbl)
            if not isinstance(node, ComputationNode):
                msg = f"Evaluation can only be performed on ComputationNodes. Received: {node}."
                raise TypeError(msg)

            # Use eval_step (if supported)
            if isinstance(node, EvaluableMixin):
                step_res = node.eval_step(batch_input=cache, losses=node_losses)

            # Use manual forward + loss comp otherwise (no optimizer contribution)
            else:
                if hasattr(node, "model") and hasattr(node.model, "eval"):
                    node.model.eval()
                model_output: BatchOutput = node.forward(node.get_input_batch(cache))
                loss_records: list[LossRecord] = []
                if node_losses is not None:
                    for loss in node_losses:
                        weighted_raw_loss = loss.compute(
                            batch_input=cache,
                            model_outputs={lbl: model_output},
                        )
                        lr = LossRecord(
                            value=weighted_raw_loss,
                            label=loss.label,
                            contributes_to_update=False,  # does not contribute to loss stepping
                        )
                        loss_records.append(lr)
                lc = LossCollection(records=loss_records)
                step_res = StepResult(losses=lc, node_outputs={lbl: model_output})

            # Cache stage outputs
            cache[lbl] = step_res.node_outputs[lbl]

            # Merge losses into a single LossCollection
            if loss_cache is None:
                loss_cache = step_res.losses
            else:
                loss_cache += step_res.losses

        return StepResult(
            losses=loss_cache,
            node_outputs={k: v for k, v in cache.items() if k in self.connected_node_labels},
        )

    # ==========================================
    # Visuallization Methods
    # ==========================================
    def visualize(self):
        """
        Visualize the current ModelGraph using Mermaid.js syntax.

        Returns:
            IPython display object containing the graph diagram.

        Note:
            - Requires a Mermaid-compatible Markdown viewer (e.g., JupyterLab or VS Code extension).
            - Uses `modularml.visualization.Visualizer`.

        """
        from modularml.visualization import Visualizer  # noqa: PLC0415

        viz = Visualizer(self)
        return viz.display(backend="mermaid")
