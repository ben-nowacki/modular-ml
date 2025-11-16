from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import SAMPLE_ID_COLUMN
from modularml.core.splitting.split_mixin import SplitMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from modularml.core.data.sample_collection import SampleCollection
    from modularml.core.graph.featureset import FeatureSet


class FeatureSetView(SplitMixin):
    """
    Logical row-indexed view over a specific collection within a FeatureSet.

    Description:
        A `FeatureSetView` provides a lightweight, index-based handle over a
        given collection inside a parent :class:`FeatureSet`. The view \
        reatins a reference to its parent FeatureSet (for context and metadata).

        The view can be materialized into a new :class:`SampleCollection` \
        or a fully realized :class:`FeatureSet`.
    """

    def __init__(self, source: FeatureSet, indices: np.ndarray, label: str):
        super().__init__()
        self.source = source
        self.indices = indices
        self.label = label

    # =====================================================
    # Basic info
    # =====================================================
    @property
    def n_samples(self) -> int:
        """Number of rows (samples) in this view."""
        return len(self.indices)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"FeatureSetView(label='{self.label}', n_samples={len(self)}, source={self.source.label})"

    def __eq__(self, other):
        """Compares source and indicies. Label is ignored."""
        if not isinstance(other, FeatureSetView):
            msg = f"Cannot compare equality between FeatureSetView and {type(other)}"
            raise TypeError(msg)

        return self.source == other.source and self.indices == other.indices

    __hash__ = None

    # =====================================================
    # Row selection & filtering
    # =====================================================
    def select_rows(
        self,
        rel_indices: Sequence[int],
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Create a new FeatureSetView derived from this one using relative indices.

        Description:
            Produces a new view referencing the same collection and parent \
            FeatureSet but restricted to a subset of rows. The provided indices \
            are interpreted *relative to this view* and mapped to *absolute* \
            indices in the underlying collection.
        """
        rel_indices = np.asarray(rel_indices)
        if rel_indices.ndim != 1:
            raise ValueError("rel_indices must be a 1D sequence of integer positions.")
        if np.any(rel_indices >= len(self.indices)):
            raise IndexError("Some relative indices exceed the size of the current view.")
        abs_indices = np.asarray(self.indices)[rel_indices]
        return FeatureSetView(
            source=self.source,
            indices=abs_indices,
            label=label or f"{self.label}_selection",
        )

    # =====================================================
    # Conversion handling
    # =====================================================
    def to_samplecollection(self) -> SampleCollection:
        """
        Materialize this view as a new SampleCollection.

        Description:
            Produces a :class:`SampleCollection` containing only the rows \
            referenced by this view. Row filtering is applied using the \
            stored index array (`self.indices`). The full domains from the \
            parent FeatureSet are retained.

        Returns:
            SampleCollection:
                A new SampleCollection containing only the requested subset of \
                rows. **Note that the returned table shares underlying buffers \
                with the source FeatureSet (no deep copy).**

        """
        from modularml.core.data.sample_collection import SampleCollection

        # 1. Row subset (store original source indices for traceability)
        sub_table = self.source.collection.table.take(pa.array(self.indices))

        # 2. Return new collection
        return SampleCollection(table=sub_table)

    def to_pandas(self) -> pd.DataFrame:
        """Converts this view into a flattened pandas DataFrame."""
        return self.to_samplecollection().to_pandas()

    def to_featureset(self) -> FeatureSet:
        """Converts this view into a FeatureSet."""
        from modularml.core.graph.featureset import FeatureSet

        return FeatureSet(
            label=self.label,
            table=self.to_samplecollection().table,
        )

    # =====================================================
    # Comparators
    # =====================================================
    def is_disjoint_with(self, other: FeatureSetView) -> bool:
        """
        Check if this view has no overlapping samples.

        Description:
            - If both views share the same source FeatureSet, comparison is based on indices.
            - If they originate from different sources, comparison falls back to `SAMPLE_ID_COLUMN` \
                to ensure identity consistency across saved or merged datasets.
        """
        if not isinstance(other, FeatureSetView):
            msg = f"Comparison only valid between FeatureSetViews, not {type(other)}"
            raise TypeError(msg)

        # Same collection (fast index-based check)
        if self.source is other.source:
            return len(np.intersect1d(self.indices, other.indices, assume_unique=True)) == 0

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.source.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
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
            return self.source.collection.table[SAMPLE_ID_COLUMN].take(pa.array(overlap)).to_pylist()

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.source.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
        return list(ids_self.intersection(ids_other))
