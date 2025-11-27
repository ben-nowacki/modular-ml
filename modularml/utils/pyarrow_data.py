import ast
from typing import Literal

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import (
    DTYPE_SUFFIX,
    FEATURES_COLUMN,
    METADATA_PREFIX,
    RAW_VARIANT,
    SAMPLE_ID_COLUMN,
    SHAPE_SUFFIX,
    TAGS_COLUMN,
    TARGETS_COLUMN,
    TRANSFORMED_VARIANT,
)
from modularml.utils.shape_utils import get_shape, shape_to_tuple


def flatten_schema(schema: pa.Schema, prefix: str = "", separator: str = "."):
    """Return flattened column paths for a (possibly nested) Arrow schema."""
    cols = []
    for field in schema:
        name = f"{prefix}{field.name}"

        if pa.types.is_struct(field.type):
            # Recurse into struct fields
            child_schema = pa.schema(field.type)
            cols.extend(flatten_schema(child_schema, prefix=name + separator))
        else:
            cols.append(name)

    return cols


def normalize_numpy_dtype_for_pyarrow_dtype(dtype) -> str:
    """
    Normalize dtype across NumPy and Arrow for consistent metadata storage.

    Returns a semantic type string like 'float32', 'int64', 'string', etc.
    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.floating):
        return f"float{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.integer):
        return f"int{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
        return "string"
    # fallback for unknown / complex types
    return str(pa.from_numpy_dtype(dtype))


def get_shape_of_pyarrow_array(arr: pa.Array, *, include_nrows: bool = True) -> tuple[int, ...]:
    """Infers the shape of a PyArrow array."""
    # Get num rows
    n_rows = len(arr)
    if n_rows == 0:
        return ()

    detected: tuple[int, ...] | None = None
    for i in range(min(n_rows, 10)):  # only sample first 10 rows for speed
        if not arr[i].is_valid:
            continue
        s = get_shape(arr[i].as_py())
        s = shape_to_tuple(s.get("__shape__") if isinstance(s, dict) else s)
        if detected is None:
            detected = s
        elif detected != s:
            detected = None
            break
    if include_nrows:
        return (n_rows, *detected) if detected else (n_rows,)

    return detected if detected else ()


def get_dtype_of_pyarrow_array(arr: pa.Array) -> str:
    """
    Infer array data type directly from contents.

    Returns:
        str: String representing the element data type. \
            For example, "float32", "int64", or "bytes".

    """

    def arrow_dtype_string(dtype: pa.DataType) -> str:
        """
        Return a precise string representation of a PyArrow DataType.

        Preserves bit width (e.g. 'float32', 'int64').
        """
        if pa.types.is_binary(dtype):
            return "binary"
        if pa.types.is_string(dtype):
            return "string"
        if pa.types.is_boolean(dtype):
            return "bool"
        if pa.types.is_floating(dtype):
            return f"float{dtype.bit_width}"
        if pa.types.is_integer(dtype):
            return f"{'int' if dtype.bit_width > 0 else 'uint'}{dtype.bit_width}"
        return str(dtype)

    t = arr.type

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
    return arrow_dtype_string(t)


def make_nested_list_type(base_type: pa.DataType, depth: int) -> pa.DataType:
    """Recursively build a nested PyArrow ListType (e.g., list<list<...<float32>>>)."""
    typ = base_type
    for _ in range(depth):
        typ = pa.list_(typ)
    return typ


def numpy_to_sample_schema_column(
    name: str,
    data: np.ndarray,
    *,
    variant: Literal["raw", "transformed"] = "raw",
    dtype: pa.DataType | None = None,
    store_shape_metadata: bool = True,
    max_list_depth: int = 2,
) -> tuple[str, pa.StructArray, dict[bytes, bytes]]:
    """
    Converts a NumPy array of any rank into a PyArrow column suitable for use in SampleCollection.

    This function automatically chooses between:
      - Nested list arrays (for rank ≤ max_list_depth)
      - Binary blobs with shape metadata (for higher-rank tensors)

    Args:
        name: Column name.
        data: NumPy array of shape (N, ...).
        variant (Literal["raw", "transformed"]): The variant to store the data under.
        dtype: Optional element dtype (defaults to `pa.from_numpy_dtype(data.dtype)`).
        store_shape_metadata: If True, returns schema metadata for shape reconstruction.
        max_list_depth: Maximum nesting depth for list-based encoding (default=2).

    Returns:
        (name, pa.Array, metadata)
            name: The same as input `name`.
            pa.Array: PyArrow array storing the tensor under the variant.
            metadata: Dict with optional shape and dtype info for schema attachment.

    Raises:
        TypeError: If `data` is not a NumPy array.
        ValueError: If `data` is empty.

    """
    if variant not in [RAW_VARIANT, TRANSFORMED_VARIANT]:
        msg = f"`variant` must be one of: {[RAW_VARIANT, TRANSFORMED_VARIANT]}"
        raise ValueError(msg)
    if not isinstance(data, np.ndarray):
        msg = f"Expected NumPy array, got {type(data)}"
        raise TypeError(msg)

    if data.ndim == 0:
        raise ValueError("Cannot store scalar arrays as tensor columns (need at least 1 dimension).")

    if dtype is None:
        dtype = pa.from_numpy_dtype(data.dtype)

    # Drop trailing singleton dimensions (e.g., (N,1) to (N,))
    if data.ndim > 1 and data.shape[-1] == 1:
        data = data.squeeze(-1)

    n_samples = data.shape[0]
    shape = data.shape[1:]

    # Case 1: small tensors (1D, 2D) -> store as nested list
    if data.ndim <= max_list_depth:
        # Build recursive list type matching rank
        element_type = pa.from_numpy_dtype(data.dtype)
        list_type = make_nested_list_type(element_type, depth=data.ndim - 1)
        array = pa.array(data.tolist(), type=list_type)

    # Case 2: high-rank tensors -> store as binary
    else:
        # Convert each row (sample) to bytes
        element_size = data[0].nbytes if isinstance(data[0], np.ndarray) else np.prod(shape) * data.itemsize
        array = pa.array([data[i].tobytes() for i in range(n_samples)], type=pa.binary(element_size))

    # Metadata for shape reconstruction
    metadata = {}
    if store_shape_metadata:
        metadata = {
            f"{name}.{variant}.{SHAPE_SUFFIX}".encode(): str(shape).encode(),
            f"{name}.{variant}.{DTYPE_SUFFIX}".encode(): normalize_numpy_dtype_for_pyarrow_dtype(data.dtype).encode(),
        }

    return name, array, metadata


def pyarrow_array_to_numpy(
    array: pa.Array,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    metadata: dict[bytes, bytes] | None = None,
) -> np.ndarray:
    """
    Reconstruct a NumPy array from an Arrow tensor column.

    Supports both list-array and binary encodings.

    Args:
        array: PyArrow Array (list, list<list<>>, or binary).
        shape: Required for binary arrays if shape cannot be inferred.
        dtype: Optional NumPy dtype. If None, inferred from the Arrow array or metadata.
        metadata: Optional schema metadata dict (e.g., table.schema.metadata) for dtype/shape lookup.

    Returns:
        np.ndarray: Reconstructed tensor.

    """
    # Infer dtype automatically
    inferred_dtype = None

    if pa.types.is_list(array.type):
        value_type = array.type
        # Recursively descend to base element type (for nested lists)
        while pa.types.is_list(value_type):
            value_type = value_type.value_type
        inferred_dtype = value_type.to_pandas_dtype()

    elif pa.types.is_binary(array.type) or pa.types.is_fixed_size_binary(array.type):
        # Try from metadata
        if metadata is not None:
            for key in metadata:
                if key.endswith(b".dtype"):
                    inferred_dtype = np.dtype(metadata[key].decode())
                    break

    # Fallback to explicit dtype or np.float32
    dtype = dtype or inferred_dtype or np.float32

    # Decode data
    if pa.types.is_list(array.type):
        # Nested list representation
        pydata = array.to_pylist()
        return np.array(pydata, dtype=dtype)

    if pa.types.is_binary(array.type) or pa.types.is_fixed_size_binary(array.type):
        # Infer shape from metadata if not provided
        if shape is None and metadata is not None:
            for key in metadata:
                if key.endswith(b".shape"):
                    shape = ast.literal_eval(metadata[key].decode())
                    break
        if shape is None:
            raise ValueError("Shape must be provided for binary-encoded tensors.")

        n_rows = len(array)
        # Convert each element back into a NumPy array
        return np.stack(
            [np.frombuffer(array[i].as_buffer(), dtype=dtype).reshape(shape) for i in range(n_rows)],
            axis=0,
        )

    msg = f"Unsupported Arrow tensor column type: {array.type}"
    raise TypeError(msg)


def build_sample_schema_table(
    *,
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    tags: dict[str, np.ndarray] | None = None,
) -> pa.Table:
    """
    Build a metadata-preserving PyArrow table compatible with `SampleCollection`.

    Description:
        Converts three dictionaries of NumPy arrays (`features`, `targets`, and `tags`) \
        into a structured Arrow table. Each domain becomes a Struct column whose fields \
        are *per-column structs* containing a single variant field: `"raw"`.

        Example layout:
        ```
        features: struct<
            voltage: struct<raw: list<float32>>,
            current: struct<raw: list<float32>>
        >
        ```

        Each leaf array is stored as a PyArrow array, and accompanying metadata \
        (shape, dtype) is attached to the table schema for full reconstruction.

        Metadata keys follow the namespaced convention:
            - `features.voltage.raw.shape`
            - `features.voltage.raw.dtype`

    Args:
        features:
            Mapping of feature names → NumPy arrays.
        targets:
            Mapping of target names → NumPy arrays.
        tags:
            Mapping of tag names → NumPy arrays.

    Returns:
        pa.Table:
            A PyArrow table with structured columns (`features`, `targets`, and \
            optionally `tags`), each holding per-column variant structs and \
            complete dtype/shape metadata.

    """

    def _encode_domain(
        domain_name: str,
        domain_dict: dict[str, np.ndarray],
        meta_prefix: str | None = None,
    ):
        """Converts dict of arrays to tuple[{col_name: pa.Array}, meta_data_dict]."""
        cols: dict[str, pa.Array] = {}
        metadata = {}

        for key, np_arr in domain_dict.items():
            name, arrow_arr, meta = numpy_to_sample_schema_column(
                name=key,
                data=np_arr,
                variant=RAW_VARIANT,
            )

            # Full col name
            col_name = f"{domain_name}.{name}.{RAW_VARIANT}"
            cols[col_name] = arrow_arr

            # Prefix metadata keys if provided
            if meta_prefix:
                for k, v in meta.items():
                    # Handle both bytes and string keys gracefully
                    base = k.decode() if isinstance(k, bytes) else str(k)
                    full = f"{meta_prefix}.{base}"
                    metadata[full.encode()] = v
            else:
                metadata.update(meta)

        return cols, metadata

    # Assume all domains have same length
    n_rows = len(next(iter(features.values())))

    # Build each domain
    feature_cols, feature_meta = _encode_domain(
        domain_name=FEATURES_COLUMN,
        domain_dict=features,
        meta_prefix=f"{METADATA_PREFIX}.{FEATURES_COLUMN}",
    )
    target_cols, target_meta = _encode_domain(
        domain_name=TARGETS_COLUMN,
        domain_dict=targets,
        meta_prefix=f"{METADATA_PREFIX}.{TARGETS_COLUMN}",
    )
    tag_cols, tag_meta = _encode_domain(
        domain_name=TAGS_COLUMN,
        domain_dict=tags or {},
        meta_prefix=f"{METADATA_PREFIX}.{TAGS_COLUMN}",
    )

    # Build flat column dict
    all_cols = {**feature_cols, **target_cols, **tag_cols}

    # Ensure table has at least 1 column
    if len(all_cols) == 0:
        # Create an "empty" placeholder column of correct row count
        all_cols = {SAMPLE_ID_COLUMN: pa.array([None] * n_rows, type=pa.string())}

    # Build table
    table = pa.table(all_cols)

    # Merge and attach metadata
    all_metadata = {**feature_meta, **target_meta, **tag_meta}
    if table.schema.metadata:
        all_metadata.update(table.schema.metadata)
    table = table.replace_schema_metadata(all_metadata)

    return table
