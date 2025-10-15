import ast

import numpy as np
import pyarrow as pa

from modularml.core.data.sample_schema import (
    DTYPE_POSTFIX,
    FEATURES_COLUMN,
    METADATA_PREFIX,
    SAMPLE_ID_COLUMN,
    SHAPE_POSTFIX,
    TAGS_COLUMN,
    TARGETS_COLUMN,
)


def make_nested_list_type(base_type: pa.DataType, depth: int) -> pa.DataType:
    """Recursively build a nested PyArrow ListType (e.g., list<list<...<float32>>>)."""
    typ = base_type
    for _ in range(depth):
        typ = pa.list_(typ)
    return typ


def numpy_to_pyarrow_column(
    name: str,
    data: np.ndarray,
    *,
    dtype: pa.DataType | None = None,
    store_shape_metadata: bool = True,
    max_list_depth: int = 2,
) -> tuple[str, pa.Array, dict[bytes, bytes]]:
    """
    Converts a NumPy array of any rank into a PyArrow column suitable for use in a Table.

    This function automatically chooses between:
      - Nested list arrays (for rank ≤ max_list_depth)
      - Binary blobs with shape metadata (for higher-rank tensors)

    Args:
        name: Column name.
        data: NumPy array of shape (N, ...).
        dtype: Optional element dtype (defaults to `pa.from_numpy_dtype(data.dtype)`).
        store_shape_metadata: If True, returns schema metadata for shape reconstruction.
        max_list_depth: Maximum nesting depth for list-based encoding (default=2).

    Returns:
        (name, pa.Array, metadata)
            name: The same as input `name`.
            pa.Array: PyArrow array storing the tensor.
            metadata: Dict with optional shape and dtype info for schema attachment.

    Raises:
        TypeError: If `data` is not a NumPy array.
        ValueError: If `data` is empty.

    """
    if not isinstance(data, np.ndarray):
        msg = f"Expected NumPy array, got {type(data)}"
        raise TypeError(msg)

    if data.ndim == 0:
        raise ValueError("Cannot store scalar arrays as tensor columns (need at least 1 dimension).")

    if dtype is None:
        dtype = pa.from_numpy_dtype(data.dtype)

    # Drop trailing singleton dimensions (e.g., (N,1) → (N,))
    if data.ndim > 1 and data.shape[-1] == 1:
        data = data.squeeze(-1)

    n_samples = data.shape[0]
    shape = data.shape[1:]

    # -------------------------------------------
    # Case 1: small tensors (1D, 2D)
    # -------------------------------------------
    if data.ndim <= max_list_depth:
        # Build recursive list type matching rank
        element_type = pa.from_numpy_dtype(data.dtype)
        list_type = make_nested_list_type(element_type, depth=data.ndim - 1)
        array = pa.array(data.tolist(), type=list_type)
    else:
        # -------------------------------------------
        # Case 2: high-rank tensors → store as binary blobs
        # -------------------------------------------
        # Convert each row (sample) to bytes
        element_size = data[0].nbytes if isinstance(data[0], np.ndarray) else np.prod(shape) * data.itemsize
        array = pa.array([data[i].tobytes() for i in range(n_samples)], type=pa.binary(element_size))

    # -------------------------------------------
    # Metadata for shape reconstruction
    # -------------------------------------------
    metadata = {}
    if store_shape_metadata:
        metadata = {
            f"{name}.{SHAPE_POSTFIX}".encode(): str(shape).encode(),
            f"{name}.{DTYPE_POSTFIX}".encode(): str(data.dtype).encode(),
        }

    return name, array, metadata


def pyarrow_column_to_numpy(
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
    tags: dict[str, np.ndarray],
) -> pa.Table:
    """
    Build a metadata-preserving PyArrow table compatible with ``SampleCollection``.

    Description:
        This function converts three groups of NumPy arrays—``features``, ``targets``,
        and ``tags``—into a single ``pyarrow.Table`` structured for use with
        ``SampleCollection``. Each group is stored as a ``StructArray`` column
        containing its respective subfields.

        During the conversion, per-array metadata (e.g., ``shape`` and ``dtype``)
        is automatically collected and attached to the resulting table schema
        under namespaced keys such as:

        - ``features.voltage.shape``
        - ``targets.soh.dtype``
        - ``tags.cell_id.shape``

        This metadata enables full reconstruction of the original NumPy arrays
        (including shape and dtype) during deserialization.

    Args:
        features:
            Dictionary mapping feature names to NumPy arrays.
            Each array must share the same first dimension (number of samples).
        targets:
            Dictionary mapping target/output names to NumPy arrays.
            Each array must share the same first dimension as ``features``.
        tags:
            Dictionary mapping metadata or identifier names to NumPy arrays
            (e.g., ``cell_id``, ``group_id``). Also must share the same
            first dimension.

    Returns:
        pa.Table:
            A PyArrow table with three struct columns—``features``, ``targets``,
            and ``tags``—plus a combined metadata dictionary containing shape
            and dtype information for all subfields.

    Notes:
        - Singleton trailing dimensions (e.g., arrays of shape ``(N, 1)``)
          are automatically squeezed to ``(N,)`` for compatibility.
        - Metadata keys are automatically prefixed using the group name
          (``features``, ``targets``, or ``tags``) to avoid collisions.
        - The resulting table schema contains all metadata entries accessible via:
          ``table.schema.metadata``

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> features = {"voltage": rng.random((5, 100)), "current": rng.random((5, 100))}
        >>> targets = {"soh": rng.random((5, 1))}
        >>> tags = {"cell_id": rng.integers(0, 5, (5, 1))}
        >>> table = build_samplecollection_table(features=features, targets=targets, tags=tags)
        >>> print(table.schema.metadata)
        {
            b'features.voltage.shape': b'(100,)',
            b'features.voltage.dtype': b'float64',
            b'targets.soh.shape': b'()',
            b'targets.soh.dtype': b'float64',
            b'tags.cell_id.shape': b'()',
            b'tags.cell_id.dtype': b'int64'
        }

    """

    def _make_struct_from_dict(
        data: dict[str, np.ndarray],
        meta_prefix: str | None = None,
    ) -> tuple[pa.StructArray, dict[bytes, bytes]]:
        """Convert a dictionary of NumPy arrays into a StructArray + metadata."""
        arrays, names = [], []
        metadata: dict[bytes, bytes] = {}
        for name, arr in data.items():
            col_name, col_array, col_meta = numpy_to_pyarrow_column(name=name, data=arr)
            arrays.append(col_array)
            names.append(col_name)

            # Prefix metadata keys if provided
            if meta_prefix:
                for k, v in col_meta.items():
                    # Handle both bytes and string keys gracefully
                    key = k.decode() if isinstance(k, bytes) else str(k)
                    prefixed_key = f"{meta_prefix}.{key}"
                    metadata[prefixed_key.encode()] = v
            else:
                metadata.update(col_meta)

        return pa.StructArray.from_arrays(arrays, names=names), metadata

    # Build struct arrays and collect metadata
    features_struct, meta_features = _make_struct_from_dict(
        features,
        meta_prefix=f"{METADATA_PREFIX}.{FEATURES_COLUMN}",
    )
    targets_struct, meta_targets = _make_struct_from_dict(
        targets,
        meta_prefix=f"{METADATA_PREFIX}.{TARGETS_COLUMN}",
    )
    tags_struct, meta_tags = _make_struct_from_dict(
        tags,
        meta_prefix=f"{METADATA_PREFIX}.{TAGS_COLUMN}",
    )

    table = pa.table(
        {
            FEATURES_COLUMN: features_struct,
            TARGETS_COLUMN: targets_struct,
            TAGS_COLUMN: tags_struct,
        },
    )

    # Merge and attach metadata
    all_metadata = {}
    all_metadata.update(meta_features)
    all_metadata.update(meta_targets)
    all_metadata.update(meta_tags)
    if table.schema.metadata is not None:
        all_metadata.update(table.schema.metadata)

    table = table.replace_schema_metadata(all_metadata)
    return table
