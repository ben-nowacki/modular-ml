from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_collection_mixin import SampleCollectionMixin
from modularml.core.data.schema_constants import DOMAIN_SAMPLE_ID
from modularml.core.splitting.split_mixin import SplitMixin

if TYPE_CHECKING:
    from modularml.core.data._data_view import DataView
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.sample_collection import SampleCollection


@dataclass(frozen=True)
class FeatureSetView(SampleCollectionMixin, SplitMixin):
    """
    Immutable row+column projection of a FeatureSet.

    Intended for inspection, export, and analysis.
    """

    source: FeatureSet
    indices: np.ndarray
    columns: list[str]
    label: str | None = None

    # ================================================
    # Constructors
    # ================================================
    @classmethod
    def from_featureset(
        cls,
        fs: FeatureSet,
        *,
        rows: np.ndarray | None = None,
        columns: list[str] | None = None,
    ) -> DataView:
        if rows is None:
            rows = np.arange(fs.n_samples)
        if columns is None:
            columns = fs.collection.get_all_keys(
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
        return cls(source=fs, indices=np.asarray(rows, dtype=int), columns=columns)

    # ================================================
    # Properties & Dunders
    # ================================================
    @property
    def n_samples(self) -> int:
        """Number of rows (samples) in this view."""
        return len(self.indices)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self):
        return f"FeatureSetView(source='{self.source.label}', n_samples={self.n_samples}, label='{self.label}')"

    def __eq__(self, other):
        """Compares source and indicies. Label is ignored."""
        if not isinstance(other, FeatureSetView):
            msg = f"Cannot compare equality between FeatureSetView and {type(other)}"
            raise TypeError(msg)

        return self.source == other.source and self.indices == other.indices

    __hash__ = None

    # ================================================
    # SampleCollectionMixin
    # ================================================
    def _resolve_caller_attributes(
        self,
    ) -> tuple[SampleCollection, list[str] | None, np.ndarray | None]:
        return (self.source.collection, self.columns, self.indices)

    # ================================================
    # Comparators
    # ================================================
    def is_disjoint_with(self, other: FeatureSetView) -> bool:
        """
        Check if this view has no overlapping samples.

        Description:
            - If both views share the same source FeatureSet, comparison is based on indices.
            - If they originate from different sources, comparison falls back to `DOMAIN_SAMPLE_ID` \
                to ensure identity consistency across saved or merged datasets.
        """
        if not isinstance(other, FeatureSetView):
            msg = f"Comparison only valid between FeatureSetViews, not {type(other)}"
            raise TypeError(msg)

        # Same collection (fast index-based check)
        if self.source is other.source:
            return len(np.intersect1d(self.indices, other.indices, assume_unique=True)) == 0

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.source.collection.table[DOMAIN_SAMPLE_ID].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.collection.table[DOMAIN_SAMPLE_ID].to_pylist()[i] for i in other.indices}
        return ids_self.isdisjoint(ids_other)

    def get_overlap_with(self, other: FeatureSetView) -> list[int]:
        """
        Get overlapping sample identifiers between two FeatureSetViews.

        Returns:
            list[str]: A list of overlapping SAMPLE_ID values.

        """
        if not isinstance(other, FeatureSetView):
            msg = f"Comparison only valid between FeatureSetViews, not {type(other)}"
            raise TypeError(msg)

        # Same collection (fast index-based check)
        if self.source is other.source:
            overlap = np.intersect1d(self.indices, other.indices, assume_unique=True)
            return self.source.collection.table[DOMAIN_SAMPLE_ID].take(pa.array(overlap)).to_pylist()

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.source.collection.table[DOMAIN_SAMPLE_ID].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.collection.table[DOMAIN_SAMPLE_ID].to_pylist()[i] for i in other.indices}
        return list(ids_self.intersection(ids_other))
