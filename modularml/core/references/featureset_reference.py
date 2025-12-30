from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from modularml.core.data.sample_shapes import SampleShapes
from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_TAGS, DOMAIN_TARGETS
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.pyarrow_data import resolve_column_selectors
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.references.data_reference import DataReference


@dataclass(frozen=True)
class FeatureSetReference(Configurable, Summarizable):
    """Declarative reference to one or more columns of a FeatureSet."""

    node: str
    features: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None

    # ==========================================
    # Validation
    # ==========================================
    def __post_init__(self):
        # Must refer to at least one column
        if not any((self.features, self.targets, self.tags)):
            raise ValueError("FeatureSetReference must specify at least one of `features`, `targets`, or `tags`.")

        # Verify node exists
        avail_nodes = ExperimentContext.available_nodes()
        if self.node not in avail_nodes:
            msg = f"Node '{self.node}' does not exist in ExperimentContext. Available: {avail_nodes}"
            raise ValueError(msg)

        # Verify node is a FeatureSet
        if not ExperimentContext.node_is_featureset(node_label=self.node):
            msg = f"Node '{self.node}' is not a FeatureSet."
            raise TypeError(msg)

        # Use regex to expand any wildcards
        all_cols = ExperimentContext.get_all_featureset_keys(featureset_label=self.node)
        selected: dict[str, Any] = resolve_column_selectors(
            all_columns=all_cols,
            features=self.features,
            targets=self.targets,
            tags=self.tags,
            include_all_if_empty=False,
        )

        object.__setattr__(self, "features", tuple(selected[DOMAIN_FEATURES]) or None)
        object.__setattr__(self, "targets", tuple(selected[DOMAIN_TARGETS]) or None)
        object.__setattr__(self, "tags", tuple(selected[DOMAIN_TAGS]) or None)

        # Final safety check
        if not any((self.features, self.targets, self.tags)):
            raise ValueError("Column selectors resolved to an empty set.")

    def __str__(self) -> str:
        """
        Return a readable string representation for console or logs.

        Example:
            FeatureSetReference(node='PulseFeatures', features='...')

        """
        attrs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None and f.name != "node_id"
        }
        attr_str = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"

    __repr__ = __str__

    def _summary_rows(self) -> list[tuple]:
        return [
            ("node", self.node),
            ("features", str(list(self.features))),
            ("targets", str(list(self.targets))),
            ("tags", str(list(self.tags))),
        ]

    # ==========================================
    # ReferenceLike Protocol
    # ==========================================
    @property
    def node_label(self) -> str:
        """Label of the ExperimentNode this reference points to."""
        return self.node

    @property
    def node_id(self) -> str:
        """ID of the ExperimentNode this reference points to."""
        return ExperimentContext._node_label_to_id[self.node_label]

    def resolve(self):
        from modularml.core.references.data_reference_group import DataReferenceGroup

        # Convert fully-qualified columns to DataReferences
        refs: list[DataReference] = []
        refs.extend(self._resolve_domain_refs(self.features))
        refs.extend(self._resolve_domain_refs(self.targets))
        refs.extend(self._resolve_domain_refs(self.tags))

        if not refs:
            raise ValueError("FeatureSetReference attributes produced no DataReferences.")
        return DataReferenceGroup.from_refs(refs)

    # ==========================================
    # Utilities
    # ==========================================
    def _resolve_domain_refs(self, domain_strs: Iterable[str]) -> list[DataReference]:
        from modularml.core.references.data_reference import DataReference

        refs = []
        for c in domain_strs:
            d, k, r = c.split(".")
            refs.append(DataReference(node=self.node, node_id=self.node_id, domain=d, key=k, rep=r))
        return refs

    def resolve_shapes(self) -> SampleShapes:
        # Convert reference group to a list of columns
        dref = self.resolve()
        columns = [f"{ref.domain}.{ref.key}.{ref.rep}" for ref in dref.refs]

        # Convert coulmns to a view over the specified FeatureSet
        fs: FeatureSet = ExperimentContext.get_node(label=self.node)
        fsv: FeatureSetView = fs.select(columns=columns)

        # Get domain shapes
        features = fsv.get_features(fmt=DataFormat.NUMPY, rep=None)
        targets = fsv.get_targets(fmt=DataFormat.NUMPY, rep=None)
        tags = fsv.get_tags(fmt=DataFormat.NUMPY, rep=None)

        return SampleShapes(
            features_shape=features.shape[1:],
            targets_shape=targets.shape[1:],
            tags_shape=tags.shape[1:],
        )

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration.

        Returns:
            dict[str, Any]: Configuration used to reconstruct this reference.

        """
        return {
            "node": self.node,
            "features": self.features,
            "targets": self.targets,
            "tags": self.tags,
        }

    @classmethod
    def from_config(cls, config: dict) -> FeatureSetReference:
        """Reconstructs the reference from config."""
        return cls(
            node=config["node"],
            features=config["features"],
            targets=config["targets"],
            tags=config["tags"],
        )
