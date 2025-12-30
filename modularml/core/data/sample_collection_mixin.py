from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
)
from modularml.utils.data.data_format import DataFormat
from modularml.utils.data.pyarrow_data import resolve_column_selectors

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from modularml.core.data.featureset import FeatureSet


class SampleCollectionMixin:
    """
    Mixin providing a uniform SampleCollection accessor API.

    Description:
        Exposes read-only access to SampleCollection-backed data for FeatureSet,
        and FeatureSetViews. The concrete class is responsible for defining
        where the SampleCollection lives and which row indices are active.

    Required hook:
        _resolve_caller_attributes() -> (SampleCollection, list[str] | None, np.ndarray | None)
    """

    # ================================================
    # Core resolution hook (must be implemented by host class)
    # ================================================
    def _resolve_caller_attributes(
        self,
    ) -> tuple[SampleCollection, list[str] | None, np.ndarray | None]:
        """Must resolve the source SampleCollection, list of columns, and row indices of the caller."""
        raise NotImplementedError

    def _resolve_collection(self) -> SampleCollection:
        collection, columns, indices = self._resolve_caller_attributes()
        if columns is not None and DOMAIN_SAMPLE_ID not in columns:
            columns.append(DOMAIN_SAMPLE_ID)

        # Column filtering
        sub_table = collection.table if columns is None else collection.table.select(columns)

        # Row filtering
        sub_table = sub_table if indices is None else sub_table.take(pa.array(indices))
        return SampleCollection(table=sub_table)

    # ================================================
    # Basic properties
    # ================================================
    @property
    def n_samples(self) -> int:
        """Total number of samples (rows)."""
        collection = self._resolve_collection()
        return collection.n_samples

    def __len__(self) -> int:
        return self.n_samples

    # ================================================
    # Key accessors
    # ================================================
    def get_feature_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the feature column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            List of all feature column names.

        """
        collection = self._resolve_collection()
        return collection.get_feature_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the target column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            List of all target column names.

        """
        collection = self._resolve_collection()
        return collection.get_target_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Retrieves the tag column names.

        Description:
            Returns keys for the specified domain with fully
            explicit control over key formatting. No implicit suffix or
            prefix behavior is applied.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            List of all tag column names.

        """
        collection = self._resolve_collection()
        return collection.get_tag_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_all_keys(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Get all column names in the SampleCollection.

        Args:
            include_rep_suffix (bool):
                Whether to include the representation suffix in the keys
                (e.g., "voltage" or "voltage.raw"). Defaults to True.
            include_domain_prefix (bool): Wether to include the domain prefix in the
                keys (e.g., "cell_id" or "tags.cell_id"). Defaults to True.

        Returns:
            list[str]:
                All unique columns in the SampleCollection.

        """
        collection = self._resolve_collection()
        return collection.get_all_keys(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Shape / dtype accessors
    # ================================================
    def get_feature_shapes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of feature columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each feature representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of feature column representation names to their shapes.

        """
        collection = self._resolve_collection()
        return collection.get_feature_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_shapes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of target columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each target representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targets")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of target column representation names to their shapes.

        """
        collection = self._resolve_collection()
        return collection.get_target_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_shapes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of tag columns to their array shapes.

        Description:
            Returns a dictionary describing the shape of each tag representation
            in the FeatureSet.
            Keys are formatted depending on the bool flags and values are tuples
            representing the array shape (e.g., `(101,)`).
            The shape does not include the number of samples dimension.

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of tag column representation names to their shapes.

        """
        collection = self._resolve_collection()
        return collection.get_tag_shapes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_feature_dtypes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of feature columns to their data types.

        Description:
            Returns a dictionary describing the data type of each feature representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "features")

        Returns:
            dict[str, str]:
                Mapping of feature column names to their data-types.

        """
        collection = self._resolve_collection()
        return collection.get_feature_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_target_dtypes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of targets columns to their data types.

        Description:
            Returns a dictionary describing the data type of each targets representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "targetss")

        Returns:
            dict[str, str]:
                Mapping of targets column names to their data-types.

        """
        collection = self._resolve_collection()
        return collection.get_target_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tag_dtypes(self, *, include_rep_suffix=True, include_domain_prefix=True):
        """
        Mapping of tag columns to their data types.

        Description:
            Returns a dictionary describing the data type of each tag representation.
            Keys are formatted depending on the bool flags and values are string-based
            data-types (e.g., "float32").

        Args:
            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw")
            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags")

        Returns:
            dict[str, str]:
                Mapping of tag column names to their data-types.

        """
        collection = self._resolve_collection()
        return collection.get_tag_dtypes(
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ================================================
    # Domain data access
    # ================================================
    def get_data(
        self,
        *,
        columns: str | list[str] | None = None,
        features: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        tags: str | list[str] | None = None,
        rep: str | None = None,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        include_domain_prefix: bool = True,
        include_rep_suffix: bool = True,
    ) -> Any:
        """
        Retrieves a subset of data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified columns (e.g. "features.voltage.raw")
                - Domain-based selectors via `features`, `targets`, and `tags`
                - Wildcards (e.g. "*.raw", "voltage.*", "*.*")
                - Automatic domain prefixing ("voltage.raw" -> "features.voltage.raw")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are stacked in the following order:
                - Domain sorted with: features -> targets -> tags -> sample_id
                - Within each domain, columns are sorted alphabetically.

        Args:
            columns (str | list[str] | None):
                Fully-qualified column names to include
                (e.g. "features.voltage.raw"). These must exactly match existing
                columns in the FeatureSet.

            features (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("voltage"),
                key/rep pairs ("voltage.raw"), or wildcards.
                The "features." prefix may be omitted.

            targets (str | list[str] | None):
                Target-domain selectors, following the same rules as `features`.

            tags (str | list[str] | None):
                Tag-domain selectors, following the same rules as `features`.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Automatically included if multiple domains are included in the
                selected columns. Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to True.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        collection = self._resolve_collection()

        # Extract real columns from collection
        all_cols: list[str] = collection.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )

        # Build final column selection (organized by domain)
        selected: dict[str, set[str]] = resolve_column_selectors(
            all_columns=all_cols,
            columns=columns,
            features=features,
            targets=targets,
            tags=tags,
            rep=rep,
            include_all_if_empty=False,
        )

        # Order and flatten columns: features -> targets -> tags
        sel_cols: list[str] = []
        for d in [DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS, DOMAIN_SAMPLE_ID]:
            if d in selected:
                sel_cols.extend(sorted(selected[d]))

        # Delegate actual data extraction
        return collection.get_columns(
            columns=sel_cols,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        features: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of feature data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified features (e.g. "voltage.raw")
                - Wildcards (e.g. "*.raw", "voltage.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            features (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("voltage"),
                key/rep pairs ("voltage.raw"), or wildcards.
                The "features." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            features=features or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        targets: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of target data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified targets (e.g. "soh.raw")
                - Wildcards (e.g. "*.raw", "soh.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            targets (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("soh"),
                key/rep pairs ("soh.raw"), or wildcards.
                The "targets." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            targets=targets or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        tags: str | list[str] | None = None,
        rep: str | None = None,
        include_domain_prefix: bool = False,
        include_rep_suffix: bool = False,
    ):
        """
        Retrieves a subset of tag data from this FeatureSet/View.

        Description:
            Data selection is performed via column-wise filtering and returned
            in the format specified by `fmt`.

            Selection supports:
                - Explicit fully-qualified tags (e.g. "group_id.raw")
                - Wildcards (e.g. "*.raw", "group_id.*", "*")
                - Optional default representation inference via `rep`

            If multiple columns are selected, and a non-keyed data formed is used,
            the columns are sorted alphabetically.

        Args:
            fmt (DataFormat, optional):
                The format of the returned data. Defaults to a dict of numpy arrays
                (i.e., each key corresponds to a single column).

            tags (str | list[str] | None):
                Feature-domain selectors. May be bare keys ("group_id"),
                key/rep pairs ("group_id.raw"), or wildcards.
                The "tags." prefix may be omitted.

            rep (str | None):
                Default representation suffix to apply when a selector omits a
                representation. Explicit representations are never overridden.

            include_domain_prefix (bool):
                Whether to include domain prefixes (e.g., "tags").
                Defaults to False.

            include_rep_suffix (bool):
                Whether to include representation suffixes (e.g., "raw").
                Automatically included if multiple representations are included in
                the selected columns. Defaults to False.

        Returns:
            Data from the specified columns, in the request DataFormat.

        """
        return self.get_data(
            tags=tags or "*",
            rep=rep,
            fmt=fmt,
            include_domain_prefix=include_domain_prefix,
            include_rep_suffix=include_rep_suffix,
        )

    # ==========================================================
    # UUID access
    # ==========================================================
    def get_sample_uuids(self, fmt: DataFormat = DataFormat.NUMPY):
        """
        Retrieve sample UUIDs in this collection.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.

        Returns:
            Sample UUIDs in the requested format.

        """
        collection = self._resolve_collection()
        return collection.get_sample_uuids(fmt=fmt)

    # ==========================================================
    # Export
    # ==========================================================
    def to_arrow(self) -> pa.Table:
        collection = self._resolve_collection()

        # Sort columns (not necessary but nicer for manual inspection)
        sorted_cols = collection.get_all_keys(
            include_domain_prefix=True,
            include_rep_suffix=True,
        )
        return collection.table.select(sorted_cols)

    def to_pandas(self) -> pd.DataFrame:
        return self.to_arrow().to_pandas()

    def to_sample_collection(self) -> SampleCollection:
        return self._resolve_collection()

    def to_featureset(self, label: str) -> FeatureSet:
        from modularml.core.data.featureset import FeatureSet

        return FeatureSet(label=label, table=self.to_arrow())
