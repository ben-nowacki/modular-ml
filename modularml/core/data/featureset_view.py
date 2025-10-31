from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_collection import _evaluate_filter_conditions
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
    Logical view over a specific collection within a FeatureSet.

    Description:
        A `FeatureSetView` provides a lightweight, index-based handle over a
        given collection (e.g., "original" or "transformed") inside a parent
        :class:`FeatureSet`. The view references:
          - its parent FeatureSet (for context and metadata), and
          - a specific SampleCollection from that FeatureSet (for data access).

        It can represent:
          - a row-based subset (via indices), or
          - a joint row + column view (via selected feature/target/tag keys).

        The view can be materialized into a new :class:`SampleCollection`
        or a fully realized :class:`FeatureSet`.
    """

    source: FeatureSet
    collection_key: str
    indices: np.ndarray
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional column filters
    feature_keys: Sequence[str] | None = None
    target_keys: Sequence[str] | None = None
    tag_keys: Sequence[str] | None = None

    # =====================================================
    # Basic info
    # =====================================================
    @property
    def collection(self) -> SampleCollection:
        """The SampleCollection this view is derived from."""
        return self.source._collections[self.collection_key]

    @property
    def n_samples(self) -> int:
        """Number of rows (samples) in this view."""
        return len(self.indices)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"FeatureSetView(label='{self.label}', collection='{self.collection_key}', n_samples={len(self)})"

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
            Produces a new view referencing the same collection and parent
            FeatureSet but restricted to a subset of rows. The provided indices
            are interpreted *relative to this view* and mapped to *absolute*
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
            collection_key=self.collection_key,
            indices=abs_indices,
            label=label or f"{self.label}_selection",
            feature_keys=self.feature_keys,
            target_keys=self.target_keys,
            tag_keys=self.tag_keys,
            metadata=dict(self.metadata),
        )

    def filter(
        self,
        **conditions: dict[str, Any | list[Any], Callable],
    ) -> FeatureSetView:
        """
        Create a filtered view derived from this FeatureSetView using logical conditions.

        Description:
            Evaluates the provided conditions within the context of this view.
            Returns a new FeatureSetView referencing only rows that satisfy all
            specified conditions. Underlying data is not copied.

        Returns:
            FeatureSetView:
                A new filtered view (subset) derived from this one.

        """
        collection = self.source._collections[self.collection_key]
        full_mask = _evaluate_filter_conditions(collection, **conditions)

        # Restrict mask to current view's indices
        local_mask = full_mask[self.indices]
        selected_rel_indices = np.where(local_mask)[0]

        # Map back to absolute indices
        selected_abs_indices = self.indices[selected_rel_indices]

        return FeatureSetView(
            source=self.source,
            collection_key=self.collection_key,
            indices=selected_abs_indices,
            label=f"{self.label}_filtered",
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
        sub_table = self.collection.table.take(pa.array(self.indices))

        # 2. Check if need to filter columns
        if not any([self.feature_keys, self.target_keys, self.tag_keys]):
            return SampleCollection(table=sub_table)

        new_cols: dict[str, pa.Array] = {}
        for domain, keys in zip(
            (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN),
            (self.feature_keys, self.target_keys, self.tag_keys),
            strict=True,
        ):
            # Current struct (as a single StructArray)
            struct_chunked = sub_table.column(domain)
            struct_arr = struct_chunked.combine_chunks() if struct_chunked.num_chunks > 1 else struct_chunked.chunk(0)

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
        if self.source is other.source and self.collection_key == other.collection_key:
            return set(self.indices.tolist()).isdisjoint(set(other.indices.tolist()))

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
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
        if self.source is other.source and self.collection_key == other.collection_key:
            overlap = set(self.indices.tolist()).intersection(set(other.indices.tolist()))
            return [self.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in overlap]

        # Otherwise compare SAMPLE_IDs
        ids_self = {self.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in self.indices}
        ids_other = {other.collection.table[SAMPLE_ID_COLUMN].to_pylist()[i] for i in other.indices}
        return list(ids_self.intersection(ids_other))
