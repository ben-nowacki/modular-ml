from __future__ import annotations

import ast
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa

from modularml.core.data.sample_schema import (
    DTYPE_SUFFIX,
    FEATURES_COLUMN,
    METADATA_PREFIX,
    METADATA_SCHEMA_VERSION_KEY,
    RAW_VARIANT,
    SAMPLE_ID_COLUMN,
    SCHEMA_VERSION,
    SHAPE_SUFFIX,
    TAGS_COLUMN,
    TARGETS_COLUMN,
    SampleSchema,
)
from modularml.utils.data_conversion import convert_dict_to_format, convert_to_format, stack_nested_numpy
from modularml.utils.data_format import DataFormat, ensure_list
from modularml.utils.pyarrow_data import (
    flatten_schema,
    get_dtype_of_pyarrow_array,
    get_shape_of_pyarrow_array,
    numpy_to_sample_schema_column,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class SampleCollection:
    """
    Arrow-backed collection of samples (features, targets, tags, and sample IDs).

    Description:
        Acts as the primary ModularML data container for datasets, batches,
        and model outputs. Data are stored in a `pyarrow.Table` whose schema
        follows the `SampleSchema` contract.

        Ensures the presence of a globally unique `sample_id` column for
        row-level traceability across all views, subsets, and exports.

    Args:
        table: Underlying Arrow table following a defined schema.
        schema: Optional `SampleSchema` describing the expected domain layout.

    """

    table: pa.Table
    schema: SampleSchema | None = None

    # =====================================================================
    # Initialization
    # =====================================================================
    def __post_init__(self):
        """
        Finalize initialization by ensuring `sample_id` exists and schema is valid.

        Description:
            - If the `sample_id` column is missing, generates a UUID for each row.
            - If duplicates are found, reassigns new UUIDs to conflicting entries.
            - Infers or validates schema consistency.
            - Embeds column shape/dtype metadata.
        """
        self._ensure_sample_id_column()
        if self.schema is None:
            self.schema = SampleSchema.from_table(self.table)
        self._validate_schema()
        self._embed_metadata()

    def _ensure_sample_id_column(self):
        """
        Ensure that the Arrow table contains a unique string-based sample ID column.

        Description:
            - If missing, adds a new column with UUID4 strings.
            - If present, checks for:
                1. String type
                2. Uniqueness across all rows
            - If duplicates are detected, regenerates new UUIDs to ensure uniqueness.
        """
        colnames = set(self.table.column_names)

        # Case 1: Add new column if missing
        if SAMPLE_ID_COLUMN not in colnames:
            n = self.table.num_rows
            ids = pa.array([str(uuid.uuid4()) for _ in range(n)], type=pa.string())
            self.table = self.table.append_column(SAMPLE_ID_COLUMN, ids)
            return

        # Case 2: Validate existing column
        col = self.table[SAMPLE_ID_COLUMN]
        if not pa.types.is_string(col.type):
            msg = f"'{SAMPLE_ID_COLUMN}' column must be of type string, got {col.type}."
            raise TypeError(msg)

        # Extract and check uniqueness
        ids = col.to_pandas().astype(str)
        n_unique = ids.nunique(dropna=False)
        if n_unique < len(ids):
            # Regenerate to ensure unique identifiers
            unique_ids = pa.array([str(uuid.uuid4()) for _ in range(len(ids))], type=pa.string())
            self.table = self.table.set_column(
                self.table.schema.get_field_index(SAMPLE_ID_COLUMN),
                SAMPLE_ID_COLUMN,
                unique_ids,
            )

    def _validate_schema(self):
        """Ensure required domains and the `sample_id` column exist."""
        unique_cols = set(self.table.column_names)

        # Check for duplicate/non-unique columns
        if len(unique_cols) != len(self.table.column_names):
            unique, counts = np.unique(np.asarray(self.table.column_names), return_counts=True)
            dups = unique[counts > 1]
            msg = f"Detected duplicate column names: {dups}"
            raise ValueError(msg)

        # Required column validation is done in SampleSchema

    def _embed_metadata(self) -> None:
        """
        Embed ModularML metadata (schema version and per-column shapes) into the Arrow table.

        Description:
            - Adds `modularml.sample.version` indicating the table schema version.
            - Stores `modularml.sample.shape.<domain>.<key>` entries for each column.

        This operation replaces the table's schema metadata in-place.
        """
        meta = dict(self.table.schema.metadata or {})

        # Table version
        meta[METADATA_SCHEMA_VERSION_KEY.encode()] = SCHEMA_VERSION.encode()

        # Domain shapes and dtypes
        for domain in [FEATURES_COLUMN, TARGETS_COLUMN] + ([TAGS_COLUMN] if self.has_tags else []):
            # Get shapes and dtypes
            d_shapes: dict[str, tuple[int, ...]] = self.get_domain_shapes(domain=domain)
            d_dtypes: dict[str, str] = self.get_domain_dtypes(domain=domain)
            if d_shapes.keys() != d_dtypes.keys():
                msg = f"`{domain}` keys differ between shapes and data types."

            # Update meta data
            for k in d_shapes:
                for s_dic, suffix in zip((d_shapes, d_dtypes), (SHAPE_SUFFIX, DTYPE_SUFFIX), strict=True):
                    meta_key = (f"{METADATA_PREFIX}.{domain}.{k}.{suffix}").encode()
                    meta_val = str(s_dic[k]).encode()
                    if meta_key in meta and meta_val != meta[meta_key]:
                        msg = (
                            f"Meta data already exist for key `{meta_key}` but it does not "
                            f"match expected value: {meta[meta_key]} != {meta_val}"
                        )
                        raise ValueError(msg)
                    meta[meta_key] = meta_val

        # Update table schema
        self.table = self.table.replace_schema_metadata(meta)

    def __eq__(self, other):
        if not isinstance(other, SampleCollection):
            msg = f"Cannot compare equality between FeatureSet and {type(other)}"
            raise TypeError(msg)

        return self.table.equals(other.table)

    __hash__ = None

    # =====================================================================
    # Core properties
    # =====================================================================
    @property
    def table_version(self) -> str | None:
        """
        Retrieve the ModularML schema version stored in the table metadata.

        Returns:
            Version string if present; otherwise ``None``.

        """
        meta = self.table.schema.metadata or {}
        val = meta.get(METADATA_SCHEMA_VERSION_KEY.encode())
        return val.decode() if val else None

    @property
    def available_domains(self) -> list[str]:
        return [FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN, SAMPLE_ID_COLUMN]

    @property
    def n_samples(self) -> int:
        """Total number of rows (samples) in the Arrow table."""
        return self.table.num_rows

    @property
    def has_tags(self) -> bool:
        """Return True if tags domain exists and is non-empty."""
        return len(self.tag_keys) > 0

    @property
    def feature_keys(self) -> list[str]:
        """List of feature column names defined in the schema."""
        return self.schema.domain_keys(FEATURES_COLUMN)

    @property
    def target_keys(self) -> list[str]:
        """List of target column names defined in the schema."""
        return self.schema.domain_keys(TARGETS_COLUMN)

    @property
    def tag_keys(self) -> list[str]:
        """List of tag column names defined in the schema."""
        return self.schema.domain_keys(TAGS_COLUMN)

    @property
    def feature_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Mapping of feature column variants to their array shapes.

        Description:
            Returns a dictionary describing the shape of each feature variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            values are tuples representing the array shape (e.g., `(101,)`). \
            The shape does not include the number of samples dimension.

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of feature column variant names to their shapes.

        Example:
        ``` python
            {
                "voltage.raw": (1000, 1),
                "voltage.transformed": (1000, 1),
                "current.raw": (1000, 1)
            }
        ```

        """
        shapes = {}
        for k in self.feature_keys:
            for v in self.get_variant_keys(FEATURES_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_shape(FEATURES_COLUMN, k, v)
        return shapes

    @property
    def target_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Mapping of target column variants to their array shapes.

        Description:
            Returns a dictionary describing the shape of each target variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            values are tuples representing the array shape (e.g., (e.g., `(1,)`). \
            The shape does not include the number of samples dimension.

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of target column variant names to their shapes.

        Example:
        ``` python
            {
                "soh.raw": (1000, 1),
                "soh.transformed": (1000, 1),
                "soc.raw": (1000, 1)
            }
        ```

        """
        shapes = {}
        for k in self.target_keys:
            for v in self.get_variant_keys(TARGETS_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_shape(TARGETS_COLUMN, k, v)
        return shapes

    @property
    def tag_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        Mapping of tag column variants to their array shapes.

        Description:
            Returns a dictionary describing the shape of each tag variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            values are tuples representing the array shape (e.g., `(1,)`). \
            The shape does not include the number of samples dimension.

        Returns:
            dict[str, tuple[int, ...]]:
                Mapping of tag column variant names to their shapes.

        Example:
        ``` python
            {
                "str_label.raw": (1000, 1),
                "str_label.transformed": (1000, 5),
                "group_id.raw": (1000, 1)
            }
        ```

        """
        if not self.has_tags:
            return {}
        shapes = {}
        for k in self.tag_keys:
            for v in self.get_variant_keys(TAGS_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_shape(TAGS_COLUMN, k, v)
        return shapes

    @property
    def feature_dtypes(self) -> dict[str, str]:
        """
        Mapping of feature column variants to their array data types.

        Description:
            Returns a dictionary describing the data type of each feature variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of feature column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "voltage.raw": "float32",
                "voltage.transformed": "float32",
            }
        ```

        """
        shapes = {}
        for k in self.feature_keys:
            for v in self.get_variant_keys(FEATURES_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_dtype(FEATURES_COLUMN, k, v)
        return shapes

    @property
    def target_dtypes(self) -> dict[str, str]:
        """
        Mapping of target column variants to their array data types.

        Description:
            Returns a dictionary describing the data type of each target variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of target column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "soh.raw": "float32",
                "soh.transformed": "float32",
            }
        ```

        """
        shapes = {}
        for k in self.target_keys:
            for v in self.get_variant_keys(TARGETS_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_dtype(TARGETS_COLUMN, k, v)
        return shapes

    @property
    def tag_dtypes(self) -> dict[str, str]:
        """
        Mapping of tag column variants to their array data types.

        Description:
            Returns a dictionary describing the data type of each tag variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of tag column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "str_label.raw": "string",
                "str_label.transformed": "int8",
            }
        ```

        """
        if not self.has_tags:
            return {}
        shapes = {}
        for k in self.tag_keys:
            for v in self.get_variant_keys(TAGS_COLUMN, k):
                shapes[f"{k}.{v}"] = self.get_variant_dtype(TAGS_COLUMN, k, v)
        return shapes

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve feature data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of feature keys to return. \
                If None, all feature keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the feature keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                feature keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                feature keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self.get_domain_data(
            domain=FEATURES_COLUMN,
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve target data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of target keys to return. \
                If None, all target keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the target keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                target keys (e.g., "soh" or "soh.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                target keys (e.g., "soh" or "targets.soh"). Defaults to False.

        Returns:
            Target data in the requested format.

        """
        return self.get_domain_data(
            domain=TARGETS_COLUMN,
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve tag data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of tag keys to return. \
                If None, all tag keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the tag keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                tag keys (e.g., "cell_id" or "cell_id.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                tag keys (e.g., "cell_id" or "tags.cell_id"). Defaults to False.

        Returns:
            Tag data in the requested format.

        """
        return self.get_domain_data(
            domain=TAGS_COLUMN,
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_sample_uuids(self, fmt: DataFormat = DataFormat.NUMPY):
        """
        Retrieve sample UUIDs in this collection.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.

        Returns:
            Sample UUIDs in the requested format.

        """
        # Access the sample_id column as an Arrow array
        pa_arr = self.table.column(SAMPLE_ID_COLUMN).combine_chunks()

        # Return raw pyarrow array is no format is specified
        if fmt is None:
            return pa_arr

        # Otherwise PyArrow's `to_numpy` for list-like arrays (1D/2D)
        # .to_numpy() returns an object-array, we need to convert to proper shape and dtype
        data = pa_arr.to_numpy(zero_copy_only=False)
        np_arr = stack_nested_numpy(data, (1,))
        return convert_to_format(np_arr, fmt=fmt)

    def get_all_keys(
        self,
        *,
        include_variant_suffix: bool = True,
        include_domain_prefix: bool = True,
    ) -> list[str]:
        """
        Get all column names in the SampleCollection.

        Args:
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                keys (e.g., "voltage" or "voltage.raw"). Defaults to True.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                keys (e.g., "cell_id" or "tags.cell_id"). Defaults to True.

        Returns:
            list[str]: All unique columns in the SampleCollection.

        """
        out = []
        for col in flatten_schema(self.table.schema, separator="."):
            # sample_id either gets dropped (if !include_domain_prefix) or returned (has no variant)
            if col == SAMPLE_ID_COLUMN:
                if include_domain_prefix:
                    out.append(col)
                continue

            # Split column by "."
            parts = col.split(sep=".")
            if len(parts) != 3:
                msg = f"Unknown column schema for col `{col}`. Should have 3 parts but detected {len(parts)}."
                raise ValueError(msg)

            # Remove specified components
            if not include_domain_prefix:
                parts = parts[1:]
            if not include_variant_suffix:
                parts = parts[:-1]

            # Re-join with "." separator
            out.append(".".join(parts))

        return sorted(set(out))

    # =====================================================================
    # Domain-level accessors
    # =====================================================================
    def get_domain_keys(self, domain: str) -> list[str]:
        if domain == FEATURES_COLUMN:
            return self.feature_keys
        if domain == TARGETS_COLUMN:
            return self.target_keys
        if domain == TAGS_COLUMN:
            return self.tag_keys
        msg = f"Invalid domain: `{domain}`"
        raise ValueError(msg)

    def get_domain_shapes(self, domain: str) -> dict[str, tuple[int, ...]]:
        if domain == FEATURES_COLUMN:
            return self.feature_shapes
        if domain == TARGETS_COLUMN:
            return self.target_shapes
        if domain == TAGS_COLUMN:
            return self.tag_shapes
        msg = f"Invalid domain: `{domain}`"
        raise ValueError(msg)

    def get_domain_dtypes(self, domain: str) -> dict[str, str]:
        if domain == FEATURES_COLUMN:
            return self.feature_dtypes
        if domain == TARGETS_COLUMN:
            return self.target_dtypes
        if domain == TAGS_COLUMN:
            return self.tag_dtypes
        msg = f"Invalid domain: `{domain}`"
        raise ValueError(msg)

    def get_domain_data(
        self,
        domain: str,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = None,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve domain data in a chosen format.

        Args:
            domain (str):
                domain (str): One of {"features", "targets", "tags"}.
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of domain keys to return. \
                If None, all feature keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the domain keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfiied, `keys` must all have matching variants. Defaults to None.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                domain keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                domain keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Domain data in the requested format.

        """
        keys = ensure_list(keys)
        domain_keys = self.get_domain_keys(domain=domain)
        invalid = set(keys).difference(set(domain_keys))
        if invalid:
            msg = f"The following keys do not exist in the `{domain}` domain: {invalid}"
            raise ValueError(msg)

        if variant is None:
            include_variant_suffix = True

        res = {}
        for k in keys or domain_keys:
            variants = ensure_list(variant) if variant else self.get_variant_keys(domain=domain, key=k)
            for v in variants:
                f_key = (
                    _ensure_domain_prefix(k, domain=domain)
                    if include_domain_prefix
                    else _remove_domain_prefix(k, domain=domain)
                )
                f_key = (
                    _ensure_variant_suffix(f_key, variant=v)
                    if include_variant_suffix
                    else _remove_variant_suffix(f_key, variant=v)
                )

                res[f_key] = self.get_variant_data(
                    domain=domain,
                    key=k,
                    variant=v,
                    fmt=DataFormat.NUMPY,
                )

        if fmt == DataFormat.DICT_NUMPY:
            return res
        return convert_dict_to_format(res, fmt=fmt, mode="stack", axis=1)

    # =====================================================================
    # Variant-level accessors
    # =====================================================================
    def _colname(self, domain: str, key: str, variant: str) -> str:
        return f"{domain}.{key}.{variant}"

    def get_variant_keys(self, domain: str, key: str) -> list[str]:
        """
        Get available variants for the specified domain and column.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.

        Returns:
            list[str]: Available variant names for the specified column.

        """
        if domain not in (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN):
            msg = f"Invalid domain '{domain}'"
            raise ValueError(msg)
        return list(self.schema.variant_keys(domain, key))

    def get_variant_shape(self, domain: str, key: str, variant: str) -> tuple[int, ...]:
        """
        Obtain the shape of a specific column and variant.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.
            variant (str): Column variant (e.g., "raw" or "transformed")

        Returns:
            tuple[int, ...]: Data shape for specific column and variant.

        """
        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        meta_key = f"{METADATA_PREFIX}.{domain}.{key}.{variant}.{SHAPE_SUFFIX}"
        shape = meta.get(meta_key.encode())
        if shape:
            return tuple(ast.literal_eval(shape.decode()))

        # 2. Infer shapes directly from stored data (slow)
        col_name = self._colname(domain=domain, key=key, variant=variant)
        variant_arr: pa.Array = self.table[col_name].combine_chunks()
        return get_shape_of_pyarrow_array(variant_arr, include_nrows=False)

    def get_variant_dtype(self, domain: str, key: str, variant: str) -> str:
        """
        Obtain the data type of a specific column and variant.

        Description:
            Attempts to read data types from table metadata. If not found, \
            falls back to inferring dtypes directly from stored data.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.
            variant (str): Column variant (e.g., "raw" or "transformed")

        Returns:
            Data type string (e.g., "float32", "int64") for specific column and variant.

        """
        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        meta_key = f"{METADATA_PREFIX}.{domain}.{key}.{variant}.{DTYPE_SUFFIX}"
        dtype = meta.get(meta_key.encode())
        if dtype:
            return dtype.decode()

        # 2. Infer data type directly from stored data (slow)
        col_name = self._colname(domain=domain, key=key, variant=variant)
        variant_arr: pa.Array = self.table[col_name].combine_chunks()
        return get_dtype_of_pyarrow_array(variant_arr)

    def get_variant_data(
        self,
        domain: str,
        key: str,
        variant: str,
        *,
        fmt: DataFormat | None = None,
    ):
        """
        Retrieve a single column of data in the chosen format.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Column name with in the specified domain.
            variant (str): Column variant (e.g., "raw" or "transformed")
            fmt (DataFormat, optional): Desired output format (see :class:`DataFormat`). \
                If None, the PyArrow view (pa.ListArray) is returned. Defaults to None.

        Returns:
            Column variant data in the requested format.

        """
        # Access the variant column as an Arrow array
        col_name = self._colname(domain=domain, key=key, variant=variant)
        if col_name not in self.table.column_names:
            msg = f"Column '{col_name}' not found in PyArrow table."
            raise KeyError(msg)
        pa_arr: pa.Array = self.table[col_name].combine_chunks()

        # Return raw pyarrow array is no format is specified
        if fmt is None:
            return pa_arr

        # Retrieve metadata for decoding
        shape = self.get_variant_shape(domain=domain, key=key, variant=variant)
        dtype_str = self.get_variant_dtype(domain=domain, key=key, variant=variant)

        # Handle binary-encoded tensors
        if pa.types.is_fixed_size_binary(pa_arr.type):
            try:
                np_dtype = np.dtype(dtype_str)
            except Exception as e:
                msg = f"Invalid dtype string '{dtype_str}' in metadata."
                raise ValueError(msg) from e

            flat_size = int(np.prod(shape))
            n_rows = len(pa_arr)

            # Decode each row's binary buffer to an ndarray of the correct dtype/shape
            decoded = np.empty((n_rows, *tuple(shape)), dtype=np_dtype)
            for i in range(n_rows):
                buf = pa_arr[i].as_buffer() if hasattr(pa_arr[i], "as_buffer") else pa_arr[i]
                if buf is None:
                    decoded[i] = np.zeros(shape, dtype=np_dtype)
                    continue
                row_bytes = np.frombuffer(buf, dtype=np_dtype, count=flat_size)
                decoded[i] = row_bytes.reshape(shape)

            return convert_to_format(decoded, fmt=fmt)

        # Otherwise PyArrow's `to_numpy` for list-like arrays (1D/2D)
        # .to_numpy() returns an object-array, we need to convert to proper shape and dtype
        data = pa_arr.to_numpy(zero_copy_only=False)
        np_arr = stack_nested_numpy(data, shape)
        return convert_to_format(np_arr, fmt=fmt)

    # =====================================================================
    # Controllled Mutation
    # =====================================================================
    # PyArrow tables are immutable, meaning any changes require a full
    # rebuild. This should be done very sparingly.
    # Current rules are:
    #   - Only variants of existing domain keys can be added/modified. Examples:
    #       - You can add/modify a "transformed" variant of an existing key
    #       - You can't add an entirely new key
    #   - The "raw" (default) variant cannot be modified
    #   - Meta data must be rebuilt for any changed columns
    #   - Any new variants must have the same number of samples
    #   - Mutations to the `table` attribute must happen inplace
    def add_variant(
        self,
        domain: str,
        key: str,
        variant: str,
        data: np.ndarray,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Add or overwrite a variant of an existing feature/target/tag key.

        Description:
            New variants can be added to an existing key. The `"raw"` (default) \
            variant cannot be overridden.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Existing key within the domain (e.g., "voltage").
            variant (str): Variant name to add or overwrite (e.g., "transformed").
            data (np.ndarray): NumPy array of shape (n_samples, ...).
            overwrite (bool, optional): Whether to replace existing variant data if \
                present. Defaults to False.

        Raises:
            ValueError: If the key does not exist or variant already exists (and overwrite=False).
            TypeError: If data is not a numpy array.

        """
        # Validate domain/key
        if key not in self.get_domain_keys(domain):
            msg = f"Cannot add variant for unknown key '{key}' in domain '{domain}'."
            raise ValueError(msg)

        # Check for valid variant
        if variant == RAW_VARIANT:
            msg = f"The default variant ('{RAW_VARIANT}') cannot be modified."
            raise ValueError(msg)

        # Check type and shape of data
        if not isinstance(data, np.ndarray):
            msg = f"Data must be a numpy array. Received: {type(data)}"
            raise TypeError(msg)
        if data.shape[0] != self.n_samples:
            msg = f"Data length {data.shape[0]} does not match number of samples {self.n_samples}."
            raise ValueError(msg)

        # Build Arrow array for the new variant column
        col_name = self._colname(domain=domain, key=key, variant=variant)
        _, new_arr, new_meta = numpy_to_sample_schema_column(
            name=key,
            data=data,
            variant=variant,
        )

        # Insert / replace column in table
        if col_name in self.table.column_names:
            if not overwrite:
                msg = f"Variant '{variant}' already exists for '{key}'. Set `overwrite=True` to modify."
                raise ValueError(msg)
            idx = self.table.schema.get_field_index(col_name)
            self.table = self.table.set_column(idx, col_name, new_arr)
        else:
            self.table = self.table.append_column(col_name, new_arr)

        # Update schema: register this variant's dtype
        if domain == FEATURES_COLUMN:
            self.schema.features.setdefault(key, {})[variant] = new_arr.type
        elif domain == TARGETS_COLUMN:
            self.schema.targets.setdefault(key, {})[variant] = new_arr.type
        else:
            self.schema.tags.setdefault(key, {})[variant] = new_arr.type

        # Update metadata
        meta = dict(self.table.schema.metadata or {})
        meta.update(new_meta)
        self.table = self.table.replace_schema_metadata(meta)

    def delete_variant(
        self,
        domain: str,
        key: str,
        variant: str,
    ) -> None:
        """
        Delete an existing variant of a feature/target/tag key.

        Description:
            The `"raw"` (default) variant cannot be deleted.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            key (str): Existing key within the domain (e.g., "voltage").
            variant (str): Variant name to add or overwrite (e.g., "transformed").

        Raises:
            ValueError: If the key does not exist or variant cannot be deleted.

        """
        # Validate domain/key
        if key not in self.get_domain_keys(domain):
            msg = f"Cannot delete variant for unknown key '{key}' in domain '{domain}'."
            raise ValueError(msg)

        # Check for valid variant
        if variant == RAW_VARIANT:
            msg = f"The default variant ('{RAW_VARIANT}') cannot be deleted."
            raise ValueError(msg)

        col_name = self._colname(domain=domain, key=key, variant=variant)
        if col_name not in self.table.column_names:
            msg = f"Column `{col_name}` does not exist in table."
            raise KeyError(msg)

        # Remove the column from the table
        idx = self.table.schema.get_field_index(col_name)
        self.table = self.table.remove_column(idx)

        # Update schema: drop this variant
        if domain == FEATURES_COLUMN:
            self.schema.features[key].pop(variant, None)
        elif domain == TARGETS_COLUMN:
            self.schema.targets[key].pop(variant, None)
        else:
            self.schema.tags[key].pop(variant, None)

        # Remove metadata entries for this variant
        meta = dict(self.table.schema.metadata or {})
        to_delete = [k for k in meta if k.decode().startswith(f"{METADATA_PREFIX}.{domain}.{key}.{variant}.")]
        for k in to_delete:
            del meta[k]
        self.table = self.table.replace_schema_metadata(meta)

    # =====================================================================
    # Export / conversion
    # =====================================================================
    def to_dict(self) -> dict[str, np.ndarray]:
        """
        Convert the entire SampleCollection into a flattened dict of NumPy arrays.

        Description:
            Returns a unified dict containing all domains (features, targets, \
            and tags) as columns. Each structured domain (e.g., features struct) \
            is flattened into individual keys named after their field keys.

        Returns:
            dict[str, np.ndarray]:
                A flattened dict containing all domains and the sample_id column.

        """
        # Gather domain data (flattened)
        all_data: dict[str, np.ndarray] = {}
        for domain in [FEATURES_COLUMN, TARGETS_COLUMN] + ([TAGS_COLUMN] if self.has_tags else []):
            d_res = self.get_domain_data(
                domain=domain,
                fmt=DataFormat.DICT_NUMPY,
                keys=None,
                variant=None,
                include_domain_prefix=True,
                include_variant_suffix=True,
            )
            all_data |= d_res

        # Always include sample_id column (guaranteed unique)
        uuids = self.get_sample_uuids(fmt=DataFormat.NUMPY)
        all_data[SAMPLE_ID_COLUMN] = stack_nested_numpy(uuids, shape=())

        return all_data

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        """
        Convert the entire SampleCollection into a flattened pandas DataFrame.

        Description:
            Returns a unified DataFrame containing all domains (features, targets, \
            and tags) as columns. Each structured domain (e.g., features struct) \
            is flattened into individual columns named after their field keys.

        Returns:
            pd.DataFrame:
                A flattened DataFrame containing all domains and the sample_id column.

        """
        # Gather domain data (flattened)
        # Each element may be a multi-dim array, need to separate 1st dimension with list()
        reshaped_data = {k: list(v) for k, v in self.to_dict().items()}
        return pd.DataFrame(reshaped_data)

    def save(self, path: str | Path) -> None:
        """
        Save SampleCollection to a single file using Arrow IPC (Feather v2).

        Description:
            This method fully fully preserves schema, domain struct columns, \
            variants, metadata (shapes/dtypes), and sample_id column.
            The `load` class method should be used for safe round-trip loading.

        Args:
            path (str | Path): Destination file path (e.g., "dataset.arrow").

        """
        path = Path(path).with_suffix(".arrow")
        with pa.OSFile(str(path), "wb") as sink:
            writer = pa.ipc.new_file(sink, self.table.schema)
            writer.write(self.table)
            writer.close()

    @classmethod
    def load(cls, path: str | Path) -> SampleCollection:
        """
        Load a SampleCollection previously saved with `.save()`.

        Description:
            All metadata, variants, domains, and sample_id column \
            are restored exactly as originally saved.

        Args:
            path (str | Path): File path to load from.

        Returns:
            SampleCollection: A fully reconstructed instance.

        """
        path = Path(path)
        with pa.OSFile(str(path), "rb") as source:
            reader = pa.ipc.open_file(source)
            table = reader.read_all()

        # Schema is inferred back from metadata inside table
        return cls(table=table)


# =========================================================================
# Helper Methods
# =========================================================================
def _remove_domain_prefix(element: str, domain: str) -> str:
    """Removes the domain prefix from element, if it contains it."""
    prefix = f"{domain}."
    return element[element.rindex(prefix) + len(prefix) :] if prefix in element else element


def _ensure_domain_prefix(element: str, domain: str) -> str:
    """Ensures the domain prefix is in element."""
    prefix = f"{domain}."
    return element if prefix in element else prefix + element


def _remove_variant_suffix(element: str, variant: str) -> str:
    """Removes the variant suffix from element, if it contains it."""
    suffix = f".{variant}"
    return element[: element.rindex(suffix)] if suffix in element else element


def _ensure_variant_suffix(element: str, variant: str) -> str:
    """Ensures the variant suffix is in element."""
    suffix = f".{variant}"
    return element if suffix in element else element + suffix


def _evaluate_single_condition(col_data: np.ndarray, cond) -> np.ndarray:
    """
    Evaluate a filter condition on a column of scalar or array values.

    Args:
        col_data (np.ndarray | Sequence[Any]):
            Column values (scalars or arrays).
        cond (Any | list | tuple | set | callable):
            The condition to evaluate:
              - callable(x) -> bool
              - iterable of allowed values
              - scalar literal for equality

    Returns:
        np.ndarray[bool]: Boolean mask indicating rows satisfying the condition.

    """
    col_data = np.asarray(col_data, dtype=object)

    # 1. Callable condition
    if callable(cond):
        # Detect scalar vs array-valued column
        first_val = col_data[0]
        if isinstance(first_val, (np.ndarray, list, tuple)):
            # Apply row-wise for arrays
            mask = np.fromiter((bool(cond(x)) for x in col_data), dtype=bool, count=len(col_data))
        else:
            # Try applying directly to the vector
            try:
                mask = np.asarray(cond(col_data))
                if np.ndim(mask) == 0:  # single scalar result
                    mask = np.full(len(col_data), bool(mask))
            except Exception as e:
                msg = f"Failed to apply callable condition. {e}"
                raise ValueError(msg) from e
        return mask

    # 2. Iterable of allowed values
    if isinstance(cond, (list, tuple, set, np.ndarray)):
        # For scalar columns -> vectorized np.isin
        first_val = col_data[0]
        if not isinstance(first_val, (np.ndarray, list, tuple)):
            return np.isin(col_data, cond, assume_unique=False)

        # Array-like row values are not supported yet
        ## This commented out below returns True is any value in the row value is in the provided
        ## condition value list. Not sure this is what is expected. Likely want an exact match?
        # # For array-valued rows -> True if *any element* in the row is in cond
        # allowed = set(cond)
        # mask = np.fromiter(
        #     (any(elem in allowed for elem in x) if x is not None else False for x in col_data),
        #     dtype=bool,
        #     count=len(col_data),
        # )
        raise TypeError("Iterable condition can only be applied to 1-dimensional columns.")

    # 3. Scalar equality condition
    # For scalar columns -> vectorized equality
    first_val = col_data[0]
    if not isinstance(first_val, (np.ndarray, list, tuple)):
        return np.asarray(col_data) == cond

    # Array-like row values are not supported yet
    ## This commented out below returns True is any value in the row value is equal to the provided
    ## condition scalar value. Not sure this is what is expected.
    # # For array-valued rows -> True if any element equals the scalar
    # mask = np.fromiter(
    #     (np.any(np.asarray(x) == cond) if x is not None else False for x in col_data),
    #     dtype=bool,
    #     count=len(col_data),
    # )
    raise TypeError("Scalar-valued conditions can only be applied to 1-dimensional columns.")


def evaluate_filter_conditions(
    collection: SampleCollection,
    variant_to_use: str = RAW_VARIANT,
    **conditions: dict[str, Any | list[Any] | Callable],
) -> np.ndarray:
    """
    Evaluate filtering conditions and return a boolean mask of matching rows.

    Used internally by both FeatureSet.filter() and FeatureSetView.filter().

    Args:
        collection (SampleCollection):
            The SampleCollection instance to filter.
        variant_to_use (str, optional):
            Conditions will only be evaluated on the specified `variant` \
            of each reference column in `conditions`. Defaults to RAW_VARIANT.
        **conditions:
                Mapping of column names (from any domain) to filter criteria.
                Values may be:
                - `scalar`: selects rows where the column equals the value.
                - `sequence`: selects rows where the column is in the given list/set.
                - `callable`: a function that takes a NumPy array and returns a boolean mask.

    Raises:
        KeyError: _description_

    Returns:
        np.ndarray:
            Mask over interal PyArrow array of samples satisfying all \
            filter conditions.

    """
    # Start with all all rows included
    mask = np.ones(collection.n_samples, dtype=bool)

    # Check available keys
    # We want to search this in tag -> feature -> target order
    ordered_domains = [
        (TAGS_COLUMN, collection.tag_keys),
        (FEATURES_COLUMN, collection.feature_keys),
        (TARGETS_COLUMN, collection.target_keys),
    ]
    all_domains = OrderedDict(ordered_domains)

    for key, cond in conditions.items():
        # Find pyarrow domain that matches this key (use first match)
        domain_of_key = next((d for d, d_keys in all_domains.items() if key in d_keys), None)
        if domain_of_key is None:
            msg = (
                f"Key '{key}' not found in features, targets, or tags. "
                f"Use `.tag_keys`, `.feature_keys`, or `.target_keys` to see available keys."
            )
            raise KeyError(msg)

        col_data = collection.get_variant_data(
            domain=domain_of_key,
            key=key,
            variant=variant_to_use,
            fmt=DataFormat.NUMPY,
        )

        cond_mask = _evaluate_single_condition(col_data=col_data, cond=cond)
        mask &= cond_mask.reshape(collection.n_samples)

    return mask
