from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.schema_constants import (
    DOMAIN_FEATURES,
    DOMAIN_SAMPLE_ID,
    DOMAIN_TAGS,
    DOMAIN_TARGETS,
    REP_RAW,
)
from modularml.utils.data.data_format import DataFormat

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
    def get_feature_keys(self, *, include_rep_suffix=False, include_domain_prefix=False):
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

    def get_target_keys(self, *, include_rep_suffix=False, include_domain_prefix=False):
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

    def get_tag_keys(self, *, include_rep_suffix=False, include_domain_prefix=False):
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
    def _get_domain(
        self,
        domain: str,
        *,
        fmt: DataFormat,
        keys=None,
        rep=REP_RAW,
        include_rep_suffix=True,
        include_domain_prefix=True,
    ):
        collection = self._resolve_collection()
        data = collection._get_domain_data(
            domain=domain,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )
        return data

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve feature data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of feature keys to return.
                If None, all feature keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the feature keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                feature keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                feature keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain(
            domain=DOMAIN_FEATURES,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve target data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of target keys to return.
                If None, all target keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the target keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                target keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                target keys (e.g., "voltage" or "targets.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain(
            domain=DOMAIN_TARGETS,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        rep: str | None = REP_RAW,
        include_rep_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve tag data in a chosen format.

        Args:
            fmt (DataFormat):
                Desired output format (see :class:`DataFormat`).
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None):
                Optional subset of tag keys to return.
                If None, all tag keys are returned. Defaults to None.
            rep (str, optional):
                The representation (e.g., "raw" or "transformed") of the tag keys to
                return. If None, all representations are returned and `include_rep_suffix` is set to True.
                If specfied, `keys` must all have matching representations. Defaults to REP_RAW.
            include_rep_suffix (bool):
                Whether to include the representation suffix in the
                tag keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool):
                Whether to include the domain prefix in the
                tag keys (e.g., "voltage" or "tags.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain(
            domain=DOMAIN_TAGS,
            fmt=fmt,
            keys=keys,
            rep=rep,
            include_rep_suffix=include_rep_suffix,
            include_domain_prefix=include_domain_prefix,
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
