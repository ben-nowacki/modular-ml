from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import ref

from matplotlib.pylab import Enum

from modularml.utils.environment import running_in_notebook

if TYPE_CHECKING:
    from modularml.components.graph_node import GraphNode
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.stage import Stage
    from modularml.core.graph.model_graph import ModelGraph
    from modularml.core.references.data_reference import DataReference


class RegistrationPolicy(Enum):
    """Controls behavior when registering objects with duplicate labels."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    NO_REGISTER = "no_register"

    @classmethod
    def from_value(cls, value):
        """Cast a string or enum value to RegistrationPolicy."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            for policy in cls:
                if policy.value == v:
                    return policy
        msg = f"Invalid registration policy: {value!r}. Expected one of: {[p.value for p in cls]}"
        raise ValueError(msg)


class ExperimentContext:
    """
    Global registry and lifecycle controller for the active Experiment.

    Notes:
        Provides a deterministic registry of:
        - GraphNodes
        - Stages
        - ModelGraphs

        By default, duplicate labels raise an error, but the behavior
        can be temporarily or permanently overridden via a
        RegistrationPolicy.

    """

    _active_exp: ref | None = None

    # Registries
    _nodes: ClassVar[dict[str, Any]] = {}
    _stages: ClassVar[dict[str, Any]] = {}
    _model_graphs: ClassVar[dict[str, Any]] = {}

    # Global policy (see _resolve_default_policy)
    _policy: ClassVar[RegistrationPolicy] | None = None

    # =====================================================
    # Policy Management
    # =====================================================
    @classmethod
    def _resolve_default_policy(cls) -> RegistrationPolicy:
        """
        Determine the default registration policy based on environment.

        Priority (highest to lowest):
            1. Environment variable
            2. Jupyter notebook detection
            3. Script default
        """
        # 1. Explicit env override
        env = os.getenv("MODULARML_EXP_REGISTRATION_POLICY")
        if env:
            return RegistrationPolicy.from_value(env)

        # 2. If running in Jupyter Notebook -> default to OVERWRITE
        if running_in_notebook():
            return RegistrationPolicy.OVERWRITE

        # 3. Else --> default to ERROR
        return RegistrationPolicy.ERROR

    @classmethod
    def _get_policy(cls) -> RegistrationPolicy:
        if cls._policy is None:
            cls._policy = cls._resolve_default_policy()
        return cls._policy

    @classmethod
    def set_registration_policy(cls, policy: str | RegistrationPolicy):
        """Permanently set the registration policy."""
        cls._policy = RegistrationPolicy.from_value(policy)

    @classmethod
    @contextmanager
    def use_policy(cls, policy: str | RegistrationPolicy):
        """Temporarily override the registration policy inside a context."""
        old_policy = cls._get_policy()
        cls._policy = RegistrationPolicy.from_value(policy)
        try:
            yield
        finally:
            cls._policy = old_policy

    @classmethod
    @contextmanager
    def dont_register(cls):
        """
        Temporarily disable ALL registration.

        GraphNodes, Stages, and ModelGraphs created inside this block
        will NOT be added to the ExperimentContext registry.

        Example:
            with ExperimentContext.dont_register():
                internal_copy = ModelStage.from_state(...)

        """
        old_policy = cls._get_policy()
        cls._policy = RegistrationPolicy.NO_REGISTER
        try:
            yield
        finally:
            cls._policy = old_policy

    @classmethod
    def with_policy(cls, policy: str | RegistrationPolicy):
        """Decorator that applies a temporary registration policy."""
        new_policy = RegistrationPolicy.from_value(policy)

        def decorator(func):
            def wrapper(*args, **kwargs):
                with cls.use_policy(new_policy):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    # =====================================================
    # Experiment activation
    # =====================================================
    @classmethod
    def activate(cls, experiment: Experiment):
        cls._active_exp = ref(experiment)
        cls.clear_registries()

    @classmethod
    def get_active(cls):
        return None if cls._active_exp is None else cls._active_exp()

    # =====================================================
    # Registration
    # =====================================================
    @classmethod
    def _handle_registration(
        cls,
        registry: dict[str, Any],
        label: str,
        obj: Any,
        *,
        obj_type: str,
    ):
        """Internal helper applying policy to registration events."""
        if cls._get_policy() is RegistrationPolicy.NO_REGISTER:
            return

        if label in registry:
            if cls._get_policy() is RegistrationPolicy.ERROR:
                msg = (
                    f"{obj_type} with label '{label}' is already registered. "
                    f"Set policy to OVERWRITE to replace existing objects."
                )
                raise ValueError(msg)

            if cls._get_policy() is RegistrationPolicy.OVERWRITE:
                registry[label] = obj
                return

            msg = f"Unknown registration policy: {cls._get_policy()}"
            raise RuntimeError(msg)

        registry[label] = obj

    @classmethod
    def register_node(cls, node: GraphNode):
        from modularml.components.graph_node import GraphNode

        if not isinstance(node, GraphNode):
            msg = f"`node` must be a GraphNode. Received: {type(node)}"
            raise TypeError(msg)
        cls._handle_registration(cls._nodes, node.label, node, obj_type="GraphNode")

    @classmethod
    def register_stage(cls, stage: Stage):
        from modularml.core.experiment.stage import Stage

        if not isinstance(stage, Stage):
            msg = f"`stage` must be a Stage. Received: {type(stage)}."
            raise TypeError(msg)
        cls._handle_registration(cls._stages, stage.label, stage, obj_type="Stage")

    @classmethod
    def register_model_graph(cls, graph: ModelGraph):
        from modularml.core.graph.model_graph import ModelGraph

        if not isinstance(graph, ModelGraph):
            msg = f"`graph` must be a ModelGraph. Received: {type(graph)}."
            raise TypeError(msg)
        cls._handle_registration(cls._model_graphs, graph.label, graph, obj_type="ModelGraph")

    @classmethod
    def clear_registries(cls):
        cls._nodes.clear()
        cls._stages.clear()
        cls._model_graphs.clear()

    # =====================================================
    # Convience Methods
    # =====================================================
    @classmethod
    def available_nodes(cls) -> list[str]:
        return list(cls._nodes.keys())

    @classmethod
    def available_stages(cls) -> list[str]:
        return list(cls._stages.keys())

    @classmethod
    def available_model_graphs(cls) -> list[str]:
        return list(cls._model_graphs.keys())

    @classmethod
    def node_is_featureset(cls, node_label: str) -> bool:
        """Checks if `node_label` is registered and is a FeatureSet."""
        from modularml.core.graph.featureset import FeatureSet

        # Check that node exists
        if node_label not in cls._nodes:
            return False

        # Check if FeatureSet
        node = cls._nodes[node_label]
        return isinstance(node, FeatureSet)

    @classmethod
    def get_all_featureset_keys(cls, featureset_label: str) -> list[str]:
        """Gets all keys belonging to the specified FeatureSet."""
        from modularml.core.graph.featureset import FeatureSet

        if featureset_label not in cls._nodes:
            msg = (
                f"FeatureSet `{featureset_label}` does not exist in this context. "
                f"Available GraphNodes: {cls.available_nodes()}"
            )
            raise ValueError(msg)
        node = cls._nodes[featureset_label]
        if not isinstance(node, FeatureSet):
            msg = f"GraphNode `{featureset_label}` is not a FeatureSet."
            raise TypeError(msg)

        return node.collection.get_all_keys(include_domain_prefix=True, include_variant_suffix=True)

    @classmethod
    def validate_data_ref(cls, ref: DataReference, *, check_featuresets: bool = True):
        """Checks that all fields of `ref` exist in the active Experiment."""
        exp = cls.get_active()

        # Check stage & node
        attrs_to_check = ["stage", "node"]
        valid_values = [cls.available_stages(), cls.available_nodes()]
        labels = ["Stage", "GraphNode"]
        for attr, allowed_values, label in zip(attrs_to_check, valid_values, labels, strict=True):
            val = getattr(ref, attr)
            if val is not None and val not in allowed_values:
                msg = (
                    f"{label} `{val}` does not exist in the active ExperimentContext. "
                    f"Available {label}s: {allowed_values}"
                )
                raise ValueError(msg)

        # Check FeatureSet fields (domain, key, variant)
        if check_featuresets:
            from modularml.core.graph.featureset import FeatureSet

            node = cls._nodes[ref.node]
            if isinstance(node, FeatureSet):
                # Check domain
                if ref.domain is not None and ref.domain not in node.collection.available_domains:
                    msg = f"DataRefBase.domain `{ref.domain}` is not a valid domain."
                    raise ValueError(msg)
                # Check key
                if ref.key is not None and ref.key not in node.collection.get_domain_keys(ref.domain):
                    msg = f"DataRefBase.key `{ref.key}` is not a valid key in domain `{ref.domain}`."
                    raise ValueError(msg)
                # Check variant
                if ref.variant is not None and ref.variant not in node.collection.get_variant_keys(ref.domain, ref.key):
                    msg = f"DataRefBase.variant `{ref.variant}` is not a valid variant for key `{ref.key}` in domain `{ref.domain}`."
                    raise ValueError(msg)
            elif ref.domain is not None or ref.key is not None or ref.variant is not None:
                msg = f"DataRefBase.node `{ref.node}` is not a FeatureSet, but defines a domain, key, or variant attribute."
                raise ValueError(msg)

        # TODO: still need to validate split, fold, role, batch
