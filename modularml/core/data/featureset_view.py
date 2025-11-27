from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import (
    FEATURES_COLUMN,
    RAW_VARIANT,
    SAMPLE_ID_COLUMN,
    TAGS_COLUMN,
    TARGETS_COLUMN,
)
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
    def to_samplecollection(
        self,
        *,
        feature_keys: list[str] | None = None,
        target_keys: list[str] | None = None,
        tag_keys: list[str] | None = None,
        variant: str | None = None,
        missing: str = "error",
    ) -> SampleCollection:
        """
        Materialize this view as a new SampleCollection.

        Description:
            Produces a :class:`SampleCollection` containing only the rows \
            referenced by this view. Row filtering is applied using the \
            stored index array (`self.indices`). The full domains from the \
            parent FeatureSet are retained.

        Args:
            feature_keys (list[str] | None):
                Only the specified feature_keys will be included in the returned
                SampleCollection. If None, all features in the original source
                SampleCollection are included. Defaults to None.
            target_keys (list[str] | None):
                Only the specified target_keys will be included in the returned
                SampleCollection. If None, all targets in the original source
                SampleCollection are included. Defaults to None.
            tag_keys (list[str] | None):
                Only the specified tag_keys will be included in the returned
                SampleCollection. If None, all tags in the original source
                SampleCollection are included. Defaults to None.
            variant (str | None):
                Primary variant to select (e.g., "transformed").
                If None, all variants are included without filtering.
                If a column for the requested variant is missing, "raw" is used.
            missing ({"error", "warn", "ignore"}):
                Behavior when a requested key or variant is missing:
                    "error" -> raise KeyError
                    "warn"  -> issue a warning and skip the key
                    "ignore"-> silently skip the key

        Returns:
            SampleCollection:
                A new SampleCollection containing only the requested subset of \
                rows and columns. **Note that the returned table shares underlying \
                buffers with the source FeatureSet (no deep copy).**

        Raises:
            KeyError:
                If missing="error" and a requested key/variant is unavailable.
            ValueError:
                For invalid missing-mode specifiers.

        """
        from modularml.core.data.sample_collection import SampleCollection

        if missing not in {"error", "warn", "ignore"}:
            msg = f"`missing` must be one of 'error', 'warn', or 'ignore'. Received: {missing}"
            raise ValueError(msg)

        # 1. Row filtering (store original source indices for traceability)
        sub_table = self.source.collection.table.take(pa.array(self.indices))

        # If no column filtering, just return
        if all(x is None for x in [feature_keys, target_keys, tag_keys, variant]):
            return SampleCollection(table=sub_table)

        # 2. Perform column filtering
        # Get all flattened column names (e.g., "features.voltage.raw"); includes SAMPLE_ID column
        col_names = self.source.collection.get_all_keys(include_domain_prefix=True, include_variant_suffix=True)

        def handle_missing(message: str):
            """Apply missing behavior."""
            if missing == "error":
                raise KeyError(message)
            if missing == "warn":
                warnings.warn(message, stacklevel=2)
            # if ignore, pass silently

        def resolve_column(domain: str, key: str, target_variant: str | None) -> list:
            """
            Resolve "{domain}.{key}.{variant}" with fallback to RAW.

            Returns a list of full column names or an empty list (if ignore/warn).
            """
            # No variant constraint -> return all variants
            if target_variant is None:
                matches = [c for c in col_names if c.startswith(f"{domain}.{key}.")]
                if not matches:
                    handle_missing(f"No columns found for key '{key}' in domain '{domain}'.")
                return matches

            # Try requested variant
            col_exact = f"{domain}.{key}.{target_variant}"
            if col_exact in col_names:
                return [col_exact]

            # Try RAW fallback
            col_raw = f"{domain}.{key}.{RAW_VARIANT}"
            if col_raw in col_names:
                # If RAW fallback was used but missing="warn", warn user
                if missing == "warn":
                    warnings.warn(
                        f"Variant '{target_variant}' not found for key '{key}'. Using RAW variant.",
                        stacklevel=2,
                    )
                return [col_raw]

            # Neither exists -> missing-key handling
            handle_missing(
                f"Key '{key}' in domain '{domain}' has no available variants (checked: '{col_exact}', '{col_raw}').",
            )
            return []  # ignored case

        # Collect selected keys
        selected_cols = []

        # Filter features
        feature_cols = []
        if feature_keys is None:
            if variant is None:
                feature_cols = [c for c in col_names if c.startswith(f"{FEATURES_COLUMN}.")]
            else:
                # Collect all keys, then resolve each with fallback
                keys = {c.split(".")[1] for c in col_names if c.startswith(f"{FEATURES_COLUMN}.")}
                feature_cols = []
                for k in keys:
                    feature_cols.extend(resolve_column(FEATURES_COLUMN, k, variant))
        else:
            feature_cols = []
            for k in feature_keys:
                feature_cols.extend(resolve_column(FEATURES_COLUMN, k, variant))

        # Filter targets
        target_cols = []
        if target_keys is None:
            if variant is None:
                target_cols = [c for c in col_names if c.startswith(f"{TARGETS_COLUMN}.")]
            else:
                keys = {c.split(".")[1] for c in col_names if c.startswith(f"{TARGETS_COLUMN}.")}
                target_cols = []
                for k in keys:
                    target_cols.extend(resolve_column(TARGETS_COLUMN, k, variant))
        else:
            target_cols = []
            for k in target_keys:
                target_cols.extend(resolve_column(TARGETS_COLUMN, k, variant))

        # Filter tags
        tag_cols = []
        if tag_keys is None:
            if variant is None:
                tag_cols = [c for c in col_names if c.startswith(f"{TAGS_COLUMN}.")]
            else:
                keys = {c.split(".")[1] for c in col_names if c.startswith(f"{TAGS_COLUMN}.")}
                tag_cols = []
                for k in keys:
                    tag_cols.extend(resolve_column(TAGS_COLUMN, k, variant))
        else:
            tag_cols = []
            for k in tag_keys:
                tag_cols.extend(resolve_column(TAGS_COLUMN, k, variant))

        selected_cols.extend(feature_cols)
        selected_cols.extend(target_cols)
        selected_cols.extend(tag_cols)
        if SAMPLE_ID_COLUMN in sub_table.column_names:
            selected_cols.append(SAMPLE_ID_COLUMN)
        selected_cols = set(selected_cols)

        # Ensure all requested columns actually exist in the table
        missing_cols = [c for c in selected_cols if c not in sub_table.column_names]
        if missing_cols:
            msg = f"The following required columns are missing after filtering: {missing_cols}"
            raise KeyError(msg)

        # 3. Create sub-table with only selected_cols
        sub_table = sub_table.select(selected_cols)

        # 4. Return new SampleCollection
        return SampleCollection(table=sub_table)

    def to_pandas(
        self,
        *,
        feature_keys: list[str] | None = None,
        target_keys: list[str] | None = None,
        tag_keys: list[str] | None = None,
        variant: str | None = None,
        missing: str = "error",
    ) -> pd.DataFrame:
        """
        Converts this view into a flattened pandas DataFrame.

        Args:
            feature_keys (list[str] | None):
                Only the specified feature_keys will be included in the returned
                pd.DataFrame. If None, all features in the original source
                pd.DataFrame are included. Defaults to None.
            target_keys (list[str] | None):
                Only the specified target_keys will be included in the returned
                pd.DataFrame. If None, all targets in the original source
                pd.DataFrame are included. Defaults to None.
            tag_keys (list[str] | None):
                Only the specified tag_keys will be included in the returned
                pd.DataFrame. If None, all tags in the original source
                pd.DataFrame are included. Defaults to None.
            variant (str | None):
                Primary variant to select (e.g., "transformed").
                If None, all variants are included without filtering.
                If a column for the requested variant is missing, "raw" is used.
            missing ({"error", "warn", "ignore"}):
                Behavior when a requested key or variant is missing:
                    "error" -> raise KeyError
                    "warn"  -> issue a warning and skip the key
                    "ignore"-> silently skip the key

        Returns:
            pd.DataFrame

        """
        return self.to_samplecollection(
            feature_keys=feature_keys,
            target_keys=target_keys,
            tag_keys=tag_keys,
            variant=variant,
            missing=missing,
        ).to_pandas()

    def to_featureset(
        self,
        *,
        feature_keys: list[str] | None = None,
        target_keys: list[str] | None = None,
        tag_keys: list[str] | None = None,
        variant: str | None = None,
        missing: str = "error",
    ) -> FeatureSet:
        """
        Converts this view into a FeatureSet.

        Args:
            feature_keys (list[str] | None):
                Only the specified feature_keys will be included in the returned
                FeatureSet. If None, all features in the original source
                FeatureSet are included. Defaults to None.
            target_keys (list[str] | None):
                Only the specified target_keys will be included in the returned
                FeatureSet. If None, all targets in the original source
                FeatureSet are included. Defaults to None.
            tag_keys (list[str] | None):
                Only the specified tag_keys will be included in the returned
                FeatureSet. If None, all tags in the original source
                FeatureSet are included. Defaults to None.
            variant (str | None):
                Primary variant to select (e.g., "transformed").
                If None, all variants are included without filtering.
                If a column for the requested variant is missing, "raw" is used.
            missing ({"error", "warn", "ignore"}):
                Behavior when a requested key or variant is missing:
                    "error" -> raise KeyError
                    "warn"  -> issue a warning and skip the key
                    "ignore"-> silently skip the key

        Returns:
            FeatureSet

        """
        from modularml.core.graph.featureset import FeatureSet

        return FeatureSet(
            label=self.label,
            table=self.to_samplecollection(
                feature_keys=feature_keys,
                target_keys=target_keys,
                tag_keys=tag_keys,
                variant=variant,
                missing=missing,
            ).table,
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
