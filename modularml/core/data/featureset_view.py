from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import (
    FEATURES_COLUMN,
    SAMPLE_ID_COLUMN,
    TAGS_COLUMN,
    TARGETS_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modularml.core.data.sample_collection import SampleCollection
    from modularml.core.pipeline.graph.featureset import FeatureSet


@dataclass
class FeatureSetView:
    """
    Logical view over a parent FeatureSet with optional row and column filtering.

    Description:
        A `FeatureSetView` provides a lightweight, index-based handle over a
        parent :class:`FeatureSet` without copying underlying Arrow data. It can
        represent either:
          - a simple row-based subset (via indices), or
          - a joint row + column view (by specifying selected feature/target/tag keys).

        The view is materialized into a :class:`SampleCollection` or
        :class:`FeatureSet` when data access or transformations are required.
    """

    source: FeatureSet
    indices: np.ndarray
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional column filters
    feature_keys: Sequence[str] | None = None
    target_keys: Sequence[str] | None = None
    tag_keys: Sequence[str] | None = None

    @property
    def n_samples(self) -> int:
        """
        Number of samples contained in this view.

        Returns:
            int: Number of selected rows.

        """
        return len(self.indices)

    def __len__(self) -> int:
        """Returns number of samples in this FeatureSetView."""
        return self.n_samples

    def __repr__(self) -> str:
        return f"FeatureSetView(label='{self.label}', n_samples={len(self)})"

    # =====================================================
    # PyArrow pass-throughs
    # =====================================================
    def select_rows(
        self,
        rel_indices: Sequence[int],
        label: str | None = None,
    ) -> FeatureSetView:
        """
        Create a new FeatureSetView derived from this one using relative indices.

        Description:
            Produces a new view referencing the same parent FeatureSet but restricted
            to a subset of the rows currently visible in this view.

            The provided indices are interpreted *relative to this view* and are
            automatically mapped to *absolute indices* in the original FeatureSet.
            This guarantees that all nested views maintain global consistency and
            traceability via the same underlying SAMPLE_ID values.

        Args:
            rel_indices (Sequence[int]):
                Row positions relative to this view (e.g., output from np.where or random splits).
            label (str | None, optional):
                Label to assign to the new view. Defaults to `"{self.label}_selection"`.

        Returns:
            FeatureSetView:
                A new FeatureSetView referencing the same parent FeatureSet but filtered
                to only the specified rows.

        """
        rel_indices = np.asarray(rel_indices)

        # Safety check
        if rel_indices.ndim != 1:
            raise ValueError("rel_indices must be a 1D sequence of integer positions.")

        if np.any(rel_indices >= len(self.indices)):
            raise IndexError("Some relative indices exceed the size of the current view.")

        # Convert relative to absolute indices
        abs_indices = np.asarray(self.indices)[rel_indices]

        # Return new view with absolute index mapping
        return FeatureSetView(
            source=self.source,
            indices=abs_indices,
            label=label or f"{self.label}_selection",
            feature_keys=self.feature_keys,
            target_keys=self.target_keys,
            tag_keys=self.tag_keys,
            metadata=dict(self.metadata),
        )

    # =====================================================
    # Conversion handling
    # =====================================================
    def to_samplecollection(self) -> SampleCollection:
        """
        Materialize this view as a new SampleCollection.

        Description:
            Produces a :class:`SampleCollection` containing only the rows and
            columns referenced by this view. Row filtering is applied using
            the stored index array (`self.indices`), while column filtering is
            controlled via optional `feature_keys`, `target_keys`, and `tag_keys`.
            If no column filters are provided, the full domains from the parent
            FeatureSet are retained.

        Returns:
            SampleCollection:
                A new SampleCollection containing only the requested subset of
                rows and columns. The returned table shares underlying buffers
                with the source FeatureSet (no deep copy).

        Raises:
            KeyError:
                If any requested feature/target/tag key is not present in the
                corresponding domain of the parent FeatureSet.

        """
        from modularml.core.data.sample_collection import SampleCollection

        # 1. Row subset (store original source indices for traceability)
        sub_table = self.source.table.take(pa.array(self.indices))

        # 2. Column filtering (if any filters are provided)
        if any([self.feature_keys, self.target_keys, self.tag_keys]):
            new_cols: dict[str, pa.Array] = {}

            for domain, keys in zip(
                (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN),
                (self.feature_keys, self.target_keys, self.tag_keys),
                strict=True,
            ):
                # Current struct (as a single StructArray)
                struct_chunked = sub_table.column(domain)
                struct_arr = (
                    struct_chunked.combine_chunks() if struct_chunked.num_chunks > 1 else struct_chunked.chunk(0)
                )

                # Keep the whole struct column as-is if no keys
                if keys is None:
                    new_cols[domain] = struct_arr
                    continue

                # Validate requested keys exist
                available = {f.name for f in struct_arr.type}
                missing = set(keys) - available
                if missing:
                    msg = f"Keys {sorted(missing)} not in domain '{domain}'. Available: {sorted(available)}"
                    raise KeyError(msg)

                # Fast path: if ordering & membership match, keep as-is
                if list(keys) == [f.name for f in struct_arr.type]:
                    new_cols[domain] = struct_arr
                    continue

                # Build a new StructArray containing only selected fields (in order)
                field_arrays = [struct_arr.field(k) for k in keys]
                new_struct = pa.StructArray.from_arrays(field_arrays, names=list(keys))
                new_cols[domain] = new_struct

            # Rebuild the table with the filtered struct columns
            sub_table = pa.table(
                {
                    FEATURES_COLUMN: new_cols[FEATURES_COLUMN],
                    TARGETS_COLUMN: new_cols[TARGETS_COLUMN],
                    TAGS_COLUMN: new_cols[TAGS_COLUMN],
                    SAMPLE_ID_COLUMN: sub_table[SAMPLE_ID_COLUMN],
                },
            )

        return SampleCollection(sub_table)

    def to_featureset(self) -> FeatureSet:
        """
        Convert the view into a fully materialized FeatureSet.

        Description:
            Creates a new :class:`FeatureSet` representing the same subset as
            this view. Column filtering is applied if `feature_keys`,
            `target_keys`, or `tag_keys` are provided.

        Returns:
            FeatureSet:
                A fully materialized FeatureSet built from the view's filtered
                SampleCollection.

        """
        from modularml.core.pipeline.graph.featureset import FeatureSet

        sc = self.to_samplecollection()
        return FeatureSet(label=self.label, table=sc.table)

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

        # Case 1: Same source -> use indices (fast)
        if self.source is other.source:
            return set(self.indices.tolist()).isdisjoint(set(other.indices.tolist()))

        # Case 2: Different sources -> use sample_id values
        ids_self = {self.source.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
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

        # Same source -> compare indices
        if self.source is other.source:
            overlap = set(self.indices.tolist()).intersection(set(other.indices.tolist()))
            return [self.source.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in overlap]

        # Different sources -> compare by sample_id values
        ids_self = {self.source.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.source.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
        return list(ids_self.intersection(ids_other))
