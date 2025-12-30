from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar
from weakref import ref

from matplotlib.pylab import Enum

from modularml.core.references.reference_like import ReferenceLike
from modularml.utils.environment.environment import running_in_notebook

if TYPE_CHECKING:
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.experiment.experiment_node import ExperimentNode
    from modularml.core.experiment.stage import Stage
    from modularml.core.references.data_reference import DataReference
    from modularml.core.topology.model_graph import ModelGraph


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
        - ExperimentNodes
        - Stages
        - ModelGraphs

        By default, duplicate labels raise an error, but the behavior
        can be temporarily or permanently overridden via a
        RegistrationPolicy.

    """

    _active_exp: ref | None = None

    # ExperimentNode registry and label-to-id mapping
    _nodes_by_id: ClassVar[dict[str, ExperimentNode]] = {}
    _node_label_to_id: ClassVar[dict[str, str]] = {}

    # Stage registry and label-to-id mapping
    _stages_by_id: ClassVar[dict[str, Stage]] = {}
    _stage_label_to_id: ClassVar[dict[str, str]] = {}

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

        ExperimentNodes, Stages, and ModelGraphs created inside this block
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
    def register_experiment_node(cls, node: ExperimentNode):
        from modularml.core.experiment.experiment_node import ExperimentNode

        if not isinstance(node, ExperimentNode):
            msg = f"`node` must be an ExperimentNode. Received: {type(node)}"
            raise TypeError(msg)

        reg_policy = cls._get_policy()
        if reg_policy is RegistrationPolicy.NO_REGISTER:
            return

        node_id = node.node_id
        label = node.label

        if node_id in cls._nodes_by_id:
            if reg_policy is RegistrationPolicy.ERROR:
                msg = f"ExperimentNode with ID '{node_id}' is already registered."
                raise ValueError(msg)
            if reg_policy is RegistrationPolicy.OVERWRITE:
                old = cls._nodes_by_id[node_id]
                cls._node_label_to_id.pop(old.label, None)
            else:
                return

        # Check label collision
        if label in cls._node_label_to_id:
            if reg_policy is RegistrationPolicy.ERROR:
                msg = f"ExperimentNode label '{label}' already exists."
                raise ValueError(msg)
            if reg_policy is RegistrationPolicy.OVERWRITE:
                old_id = cls._node_label_to_id[label]
                cls._nodes_by_id.pop(old_id, None)

        # Register unique node UUID and string-based label
        cls._nodes_by_id[node.node_id] = node
        cls._node_label_to_id[node.label] = node.node_id

    @classmethod
    def update_node_label(cls, node: ExperimentNode, new_label: str):
        if new_label in cls._node_label_to_id:
            msg = f"Label '{new_label}' already exists."
            raise ValueError(msg)

        # Remove old label
        cls._node_label_to_id.pop(node.label, None)

        # Add new label
        cls._node_label_to_id[new_label] = node.node_id

    @classmethod
    def clear_registries(cls):
        cls._nodes_by_id.clear()
        cls._node_label_to_id.clear()

    # =====================================================
    # Node Lookup
    # =====================================================
    @classmethod
    def has_node(cls, *, node_id: str | None = None, label: str | None = None) -> bool:
        if node_id is not None:
            return node_id in cls._nodes_by_id
        if label is not None:
            return label in cls._node_label_to_id
        raise ValueError("Must provide `node_id` or `label`.")

    @classmethod
    def get_node(
        cls,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> ExperimentNode:
        if node_id is not None:
            try:
                return cls._nodes_by_id[node_id]
            except KeyError as exc:
                msg = f"No ExperimentNode with id '{node_id}'"
                raise KeyError(msg) from exc

        if label is not None:
            try:
                return cls._nodes_by_id[cls._node_label_to_id[label]]
            except KeyError as exc:
                msg = f"No ExperimentNode with label '{label}'"
                raise KeyError(msg) from exc

        raise ValueError("Must provide node_id or label.")

    # =====================================================
    # Convienence Methods
    # =====================================================
    @classmethod
    def available_nodes(cls) -> list[str]:
        """List of available ExperimentNode labels."""
        return list(cls._node_label_to_id.keys())

    @classmethod
    def available_stages(cls) -> list[str]:
        """List of available Stage labels."""
        return list(cls._stage_label_to_id.keys())

    @classmethod
    def node_is_featureset(cls, node_label: str) -> bool:
        """Checks if `node_label` is registered and is a FeatureSet."""
        from modularml.core.data.featureset import FeatureSet

        # Check that node exists
        try:
            node = cls.get_node(label=node_label)
        except (ValueError, KeyError):
            return False

        # Check if FeatureSet
        return isinstance(node, FeatureSet)

    @classmethod
    def get_all_featureset_keys(cls, featureset_label: str) -> list[str]:
        """Gets all keys belonging to the specified FeatureSet."""
        from modularml.core.data.featureset import FeatureSet

        if not cls.node_is_featureset(node_label=featureset_label):
            msg = (
                f"No FeatureSet exists in this context with label '{featureset_label}'. "
                f"Available ExperimentNodes: {cls.available_nodes()}"
            )
            raise ValueError(msg)

        node = cls.get_node(label=featureset_label)
        if not isinstance(node, FeatureSet):
            msg = f"ExperimentNode `{featureset_label}` is not a FeatureSet."
            raise TypeError(msg)

        return node.get_all_keys(include_domain_prefix=True, include_rep_suffix=True)

    # =====================================================
    # Resolving Nodes from ReferenceLike Objects
    # =====================================================
    @classmethod
    def _normalize_to_single_node_id(
        cls,
        ref: ReferenceLike,
    ) -> str:
        """
        Normalize a ReferenceLike object to a single ExperimentNode node_id.

        Raises:
            TypeError:
                If `ref` is not a ReferenceLike object.
            ValueError:
                If the reference group points to multiple nodes.

        """
        if not isinstance(ref, ReferenceLike):
            msg = f"Expected ReferenceLike, received {type(ref)!r}"
            raise TypeError(msg)

        node_ids = {r.node_id for r in ref.resolve().refs}
        if len(node_ids) != 1:
            msg = f"All DataReferences must resolve to the same node_id. Received: {sorted(node_ids)}"
            raise ValueError(msg)

        return next(iter(node_ids))

    @classmethod
    def has_node_for_ref(cls, ref: ReferenceLike) -> bool:
        """
        Checks if a ExperimentNode exists for the given ReferenceLike object.

        Returns:
            True if ExperimentNode exists, false otherwise.

        """
        try:
            node_id = cls._normalize_to_single_node_id(ref)
        except (TypeError, ValueError):
            return False

        return node_id in cls._nodes_by_id

    @classmethod
    def resolve_node_from_ref(cls, ref: ReferenceLike) -> ExperimentNode:
        """
        Resolve a ReferenceLike object to its corresponding ExperimentNode.

        Raises:
            TypeError:
                If `ref` is not a ReferenceLike object.
            ValueError:
                If the reference group spans multiple nodes.
            KeyError:
                If the referenced ExperimentNode does not exist in the ExperimentContext.

        """
        node_id = cls._normalize_to_single_node_id(ref)
        try:
            return cls._nodes_by_id[node_id]
        except KeyError as exc:
            msg = f"ExperimentNode  with id '{node_id}' referenced by {ref!r} does not exist in the current ExperimentContext."
            raise KeyError(msg) from exc

    # =====================================================
    # Validate DataReference Attributes
    # =====================================================
    @classmethod
    def validate_data_ref(cls, ref: DataReference, *, check_featuresets: bool = True):
        """Checks that all fields of `ref` exist in the active Experiment."""
        # exp = cls.get_active()

        # Check stage & node
        attrs_to_check = ["stage", "node"]
        valid_values = [cls.available_stages(), cls.available_nodes()]
        labels = ["Stage", "ExperimentNode"]
        for attr, allowed_values, label in zip(attrs_to_check, valid_values, labels, strict=True):
            val = getattr(ref, attr)
            if val is not None and val not in allowed_values:
                msg = (
                    f"{label} `{val}` does not exist in the active ExperimentContext. "
                    f"Available {label}s: {allowed_values}"
                )
                raise ValueError(msg)

        # Check FeatureSet fields (domain, key, rep)
        if check_featuresets:
            from modularml.core.data.featureset import FeatureSet

            node = cls.get_node(node_id=ref.node_id)
            if isinstance(node, FeatureSet):
                # Check domain
                if ref.domain is not None and ref.domain not in node.collection.available_domains:
                    msg = f"DataRefBase.domain `{ref.domain}` is not a valid domain."
                    raise ValueError(msg)
                # Check key
                if ref.key is not None and ref.key not in node.collection._get_domain_keys(
                    ref.domain,
                    include_domain_prefix=False,
                    include_rep_suffix=False,
                ):
                    msg = f"DataRefBase.key `{ref.key}` is not a valid key in domain `{ref.domain}`."
                    raise ValueError(msg)
                # Check rep
                if ref.rep is not None and ref.rep not in node.collection._get_rep_keys(ref.domain, ref.key):
                    msg = f"DataRefBase.rep `{ref.rep}` is not a valid representation for key `{ref.key}` in domain `{ref.domain}`."
                    raise ValueError(msg)
            elif ref.domain is not None or ref.key is not None or ref.rep is not None:
                msg = f"DataRefBase.node `{ref.node}` is not a FeatureSet, but defines a domain, key, or representation attribute."
                raise ValueError(msg)

        # TODO: still need to validate split, fold, role, batch
