from __future__ import annotations

import ast
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from modularml.components.shape_spec import ShapeSpec
from modularml.core.data.sample_schema import (
    DTYPE_POSTFIX,
    FEATURES_COLUMN,
    METADATA_PREFIX,
    METADATA_SCHEMA_VERSION_KEY,
    SAMPLE_ID_COLUMN,
    SCHEMA_VERSION,
    SHAPE_POSTFIX,
    TAGS_COLUMN,
    TARGETS_COLUMN,
    SampleSchema,
)
from modularml.utils.data_conversion import convert_dict_to_format
from modularml.utils.data_format import DataFormat, ensure_list, normalize_format
from modularml.utils.shape_utils import get_shape, shape_to_tuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa


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
        table: Underlying Arrow table containing structured columns:
            - FEATURES_COLUMN:  struct<...>
            - TARGETS_COLUMN:   struct<...>
            - TAGS_COLUMN:      struct<...>
            - SAMPLE_ID_COLUMN: struct<...>
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

    # =====================================================================
    # Core properties
    # =====================================================================
    @property
    def n_samples(self) -> int:
        """Total number of rows (samples) in the Arrow table."""
        return self.table.num_rows

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

    # =====================================================================
    # Metadata embedding and access
    # =====================================================================
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

        # Column shapes
        for domain in (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN):
            # Check shape metadata
            shapes = self.column_shapes(domain)
            for key, shape in shapes.items():
                meta_key = (f"{METADATA_PREFIX}.{domain}.{key}.{SHAPE_POSTFIX}").encode()
                meta_val = str(shape).encode()
                if meta_key in meta and meta_val != meta[meta_key]:
                    msg = (
                        f"Meta data already exist for key `{meta_key}` but it does not "
                        f"match expected value: {meta[meta_key]} != {meta_val}"
                    )
                    raise ValueError(msg)
                meta[meta_key] = meta_val

            # Check dtype metadata
            dtypes = self.column_dtypes(domain)
            for key, dtype in dtypes.items():
                meta_key = (f"{METADATA_PREFIX}.{domain}.{key}.{DTYPE_POSTFIX}").encode()
                meta_val = str(dtype).encode()
                if meta_key in meta and meta_val != meta[meta_key]:
                    msg = (
                        f"Meta data already exist for key `{meta_key}` but it does not "
                        f"match expected value: {meta[meta_key]} != {meta_val}"
                    )
                    raise ValueError(msg)
                meta[meta_key] = meta_val

        # Update table schema
        self.table = self.table.replace_schema_metadata(meta)

    def table_version(self) -> str | None:
        """
        Retrieve the ModularML schema version stored in the table metadata.

        Returns:
            Version string if present; otherwise ``None``.

        """
        meta = self.table.schema.metadata or {}
        val = meta.get(METADATA_SCHEMA_VERSION_KEY.encode())
        return val.decode() if val else None

    # =====================================================================
    # Validation
    # =====================================================================
    def _validate_schema(self):
        """Ensure required domains and the `sample_id` column exist."""
        expected = {SAMPLE_ID_COLUMN, FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN}
        actual = set(self.table.column_names)
        missing = expected - actual
        if missing:
            msg = f"Missing expected columns: {sorted(missing)}"
            raise ValueError(msg)

    # =====================================================================
    # Domain accessors
    # =====================================================================
    def get_features(self, fmt=DataFormat.DICT_NUMPY, *, keys=None):
        """
        Retrieve feature data in a chosen format.

        Args:
            fmt: Desired output format (see :class:`DataFormat`).
            keys: Optional subset of feature keys to return.

        Returns:
            Feature data in the requested format.

        """
        return self._get_domain(FEATURES_COLUMN, fmt=fmt, keys=keys)

    def get_targets(self, fmt=DataFormat.DICT_NUMPY, *, keys=None):
        """
        Retrieve target data in a chosen format.

        Args:
            fmt: Desired output format (see :class:`DataFormat`).
            keys: Optional subset of target keys to return.

        Returns:
            Target data in the requested format.

        """
        return self._get_domain(TARGETS_COLUMN, fmt=fmt, keys=keys)

    def get_tags(self, fmt=DataFormat.DICT_LIST, *, keys=None):
        """
        Retrieve tag data in a chosen format.

        Args:
            fmt: Desired output format (see :class:`DataFormat`).
            keys: Optional subset of tag keys to return.

        Returns:
            Tag data in the requested format.

        """
        return self._get_domain(TAGS_COLUMN, fmt=fmt, keys=keys)

    # =====================================================================
    # Shape utilities
    # =====================================================================
    def column_shapes(self, domain: str) -> dict[str, tuple[int, ...]]:
        """
        Obtain per-column shapes for a given domain.

        Description:
            Attempts to read shapes from table metadata. If not found,
            falls back to inferring shapes directly from stored data.

        Args:
            domain: One of {"features", "targets", "tags"}.

        Returns:
            Mapping of column name -> shape tuple.

        """
        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        shapes: dict[str, tuple[int, ...]] = {}

        req_prefix = f"{METADATA_PREFIX}.{domain}."
        req_postfix = f".{SHAPE_POSTFIX}"
        for k, v in meta.items():
            k_str = k.decode()
            if k_str.startswith(req_prefix) and k_str.endswith(req_postfix):
                name = k_str[len(req_prefix) : -len(req_postfix)]
                shapes[name] = ast.literal_eval(v.decode())
        if shapes:
            return shapes

        # 2. Infer shapes directly from stored data (recursive and slow)
        return self._infer_shapes_from_data(domain)

    def _infer_shapes_from_data(self, domain: str) -> dict[str, tuple[int, ...]]:
        """
        Infer column shapes directly from Arrow array contents.

        Args:
            domain: Domain to inspect.

        Returns:
            Mapping of column name -> shape tuple inferred from sample data.

        """
        # Iterate over each column
        struct_arr = self._struct_array(domain)
        result: dict[str, tuple[int, ...]] = {}
        for fld in struct_arr.type:
            col = struct_arr.field(fld.name)
            result[fld.name] = self._arrow_column_shape(col)
        return result

    def _arrow_column_shape(self, column: pa.Array) -> tuple[int, ...]:
        """
        Infer the shape of values stored in a given Arrow column.

        Args:
            column: Arrow array to inspect.

        Returns:
            Tuple representing array shape, prefixed with number of rows.

        """
        # Get num rows
        n = len(column)
        if n == 0:
            return (0,)

        # Get data shape
        detected: tuple[int, ...] | None = None
        for i in range(min(n, 10)):  # sample first few rows for speed
            if not column[i].is_valid:
                continue
            val = column[i].as_py()
            s = get_shape(val)
            s = shape_to_tuple(s.get("__shape__") if isinstance(s, dict) else s)
            if detected is None:
                detected = s
            elif detected != s:
                detected = None
                break
        return (n, *detected) if detected else (n,)

    # =====================================================================
    # Data type utilities
    # =====================================================================
    def column_dtypes(self, domain: str) -> dict[str, str]:
        """
        Obtain per-column data types for a given domain.

        Description:
            Attempts to read data types from table metadata. If not found,
            falls back to inferring dtypes directly from stored data.

        Args:
            domain: One of {"features", "targets", "tags"}.

        Returns:
            Mapping of column name -> dtype string (e.g., "float32", "int64").

        """
        # 1. Check meta-data
        meta = self.table.schema.metadata or {}
        dtypes: dict[str, str] = {}

        req_prefix = f"{METADATA_PREFIX}.{domain}."
        req_postfix = f".{DTYPE_POSTFIX}"
        for k, v in meta.items():
            k_str = k.decode()
            if k_str.startswith(req_prefix) and k_str.endswith(req_postfix):
                name = k_str[len(req_prefix) : -len(req_postfix)]
                dtypes[name] = v.decode()
        if dtypes:
            return dtypes

        # 2. Infer dtypes directly from stored data (recursive and slow)
        return self._infer_dtypes_from_data(domain)

    def _infer_dtypes_from_data(self, domain: str) -> dict[str, str]:
        """
        Infer column data types directly from Arrow array contents.

        Description:
            When metadata is missing, this function inspects the underlying
            Arrow arrays to infer element data types. The inference follows:
                - For numeric list arrays: dtype of leaf value type.
                - For binary-encoded tensors: dtype is unknown ("bytes").
                - For primitive arrays: Arrow's native type name.

        Args:
            domain: Domain to inspect (one of "features", "targets", "tags").

        Returns:
            Mapping of column name -> inferred dtype string.

        """
        struct_arr = self._struct_array(domain)
        result: dict[str, str] = {}
        for fld in struct_arr.type:
            col = struct_arr.field(fld.name)
            result[fld.name] = self._arrow_column_dtype(col)
        return result

    def _arrow_column_dtype(self, column: pa.Array) -> str:
        """
        Infer the data type of values stored in a given Arrow column.

        Description:
            Attempts to resolve the leaf value type for list arrays or
            returns the primitive Arrow type for flat columns.

        Args:
            column: PyArrow array to inspect.

        Returns:
            str: String representing the element data type.
                 For example, "float32", "int64", or "bytes".

        """
        t = column.type

        # Case 1: nested list (tensor-like)
        if pa.types.is_list(t):
            base = t
            while pa.types.is_list(base):
                base = base.value_type
            return base.to_pandas_dtype().__name__

        # Case 2: binary blobs
        if pa.types.is_binary(t) or pa.types.is_fixed_size_binary(t):
            return "bytes"

        # Case 3: primitive (int, float, string, etc.)
        return str(t)

    # =====================================================================
    # Domain extraction
    # =====================================================================
    def _get_domain(self, domain: str, *, fmt: DataFormat | str, keys: Sequence[str] | None):
        """
        Extract a specific domain and convert it to the requested format.

        Args:
            domain (str): Domain name ("features", "targets", or "tags").
            fmt (DataFormat | str): Desired output format identifier.
            keys (Sequence[str] | None): Optional subset of domain columns to select.

        Returns:
            Domain data converted to the requested format.

        """
        fmt = normalize_format(fmt)
        df = self._domain_dataframe(domain, keys)
        if fmt == DataFormat.PANDAS:
            return df
        data_dict = df.to_dict(orient="list")

        # If converting to a tensor-like object, we want to stack along the 1st axis
        # Eg {"A":(100, 5), "B":(100, 5)} -> shape = (100, *2*, 5)
        return convert_dict_to_format(
            data=data_dict,
            fmt=fmt,
            mode="stack",
            axis=1,
            align_singletons=True,
        )

    def _domain_dataframe(self, domain: str, keys: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Convert a domain from the Arrow table into a pandas DataFrame.

        Args:
            domain (str): One of {"features", "targets", "tags"}.
            keys (Sequence[str] | None): Optional subset of domain columns to select.

        Returns:
            Flattened pandas DataFrame with one column per selected key.

        Raises:
            ValueError: If the domain name is invalid.
            KeyError: If any requested keys are missing.

        """
        keys = ensure_list(keys)

        if domain not in (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN):
            msg = f"Invalid domain '{domain}'"
            raise ValueError(msg)
        tbl = self.table.select([domain]).flatten()
        df: pd.DataFrame = tbl.to_pandas()

        # Strip "{domain}." prefix for column names (added during .flatten)
        colmap = {c: c.split(".")[-1] for c in df.columns}
        df.rename(columns=colmap, inplace=True)

        # Filter to keys, if provided
        if keys:
            missing = [k for k in keys if k not in df.columns]
            if missing:
                msg = f"Keys {missing} not in domain '{domain}'"
                raise KeyError(msg)
            df = df[keys]

        return df

    def _struct_array(self, domain: str) -> pa.StructArray:
        """
        Retrieve the Arrow StructArray corresponding to a domain.

        Args:
            domain: One of {"features", "targets", "tags"}.

        Returns:
            StructArray combining all chunks for the domain column.

        """
        col = self.table.column(domain)
        if col.num_chunks == 1:
            return col.chunk(0)
        return col.combine_chunks()

    # =====================================================================
    # Export / conversion
    # =====================================================================
    def to_table(self, domain: str | None = None) -> pa.Table:
        """
        Return the full Arrow table or a specific domain subset.

        Args:
            domain: Optional domain name. If ``None``, returns the entire table.

        Returns:
            Arrow table containing selected data.

        """
        if domain is None:
            return self.table
        return self.table.select([domain])

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the entire table into a flattened pandas DataFrame.

        Returns:
            Flattened DataFrame containing all domains.

        """
        return self.table.flatten().to_pandas()
