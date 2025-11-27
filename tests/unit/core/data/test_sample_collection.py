import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_schema import (
    FEATURES_COLUMN,
    METADATA_SCHEMA_VERSION_KEY,
    SAMPLE_ID_COLUMN,
    SCHEMA_VERSION,
)
from modularml.utils.data_format import DataFormat
from modularml.utils.pyarrow_data import build_sample_schema_table
from tests.shared.data_utils import generate_dummy_data


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def simple_pyarrow_table():
    """Builds a small but valid Arrow table with features, targets, and tags."""
    n = 5
    features = {
        "voltage": generate_dummy_data((n, 4), "float"),
        "current": generate_dummy_data((n, 4), "float"),
    }
    targets = {"soh": generate_dummy_data((n, 1), "float")}
    tags = {"cell_id": generate_dummy_data(shape=(n, 1), dtype="str", choices=["A", "B", "C", "D", "E"])}
    return build_sample_schema_table(features=features, targets=targets, tags=tags)


@pytest.fixture
def sample_collection(simple_pyarrow_table) -> SampleCollection:
    """Instantiate a SampleCollection from a valid table."""
    return SampleCollection(simple_pyarrow_table)


# ---------------------------------------------------------------------
# Basic initialization tests
# ---------------------------------------------------------------------
def test_init_and_metadata(sample_collection: SampleCollection):
    """Validate initialization, schema inference, and metadata embedding."""
    coll = sample_collection
    meta = coll.table.schema.metadata
    assert isinstance(coll, SampleCollection)
    assert coll.n_samples == 5

    # Metadata version
    assert METADATA_SCHEMA_VERSION_KEY.encode() in meta
    assert meta[METADATA_SCHEMA_VERSION_KEY.encode()].decode() == SCHEMA_VERSION

    # Check UUID column
    assert SAMPLE_ID_COLUMN in coll.table.column_names
    ids = coll.table[SAMPLE_ID_COLUMN].to_pylist()
    assert len(set(ids)) == len(ids) == coll.n_samples


def test_optional_tags_are_supported():
    """Ensure SampleCollection works even with no tags."""
    # Build table without tags
    n = 5
    features = {"f": generate_dummy_data((n, 2), dtype="float")}
    targets = {"y": generate_dummy_data((n, 1), dtype="float")}
    table_no_tags = build_sample_schema_table(features=features, targets=targets, tags=None)
    coll = SampleCollection(table_no_tags)

    assert not coll.has_tags
    assert coll.tag_keys == []


# ---------------------------------------------------------------------
# Accessor methods
# ---------------------------------------------------------------------
def test_get_features_targets_tags(sample_collection: SampleCollection):
    """Ensure get_* methods return proper numpy data."""
    feats = sample_collection.get_features(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_variant_suffix=False,
    )
    assert "voltage" in feats
    feats = sample_collection.get_features(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_variant_suffix=True,
    )
    assert "voltage.raw" in feats

    v = feats["voltage.raw"]
    assert isinstance(v, np.ndarray)
    assert v.shape == (sample_collection.n_samples, 4)

    targs = sample_collection.get_targets(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_variant_suffix=True,
    )
    assert "soh.raw" in targs
    y = targs["soh.raw"]
    assert y.shape == (sample_collection.n_samples,)

    tags = sample_collection.get_tags(
        fmt=DataFormat.DICT_NUMPY,
        include_domain_prefix=False,
        include_variant_suffix=True,
    )
    assert "cell_id.raw" in tags


def test_get_variant_data_numpy(sample_collection: SampleCollection):
    """Directly retrieve a single variant as NumPy array."""
    arr = sample_collection.get_variant_data(
        domain="features",
        key="voltage",
        variant="raw",
        fmt=DataFormat.NUMPY,
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (sample_collection.n_samples, 4)


def test_domain_accessors_are_valid(sample_collection: SampleCollection):
    """Domain keys, dtypes, and shapes should be consistent."""
    fkeys = sample_collection.feature_keys
    tkeys = sample_collection.target_keys
    assert isinstance(fkeys, list)
    assert all(isinstance(k, str) for k in fkeys)

    fshapes = sample_collection.feature_shapes
    tdtypes = sample_collection.target_dtypes
    for shape in fshapes.values():
        assert isinstance(shape, tuple)
        assert all(isinstance(x, int) for x in shape)
    for dtype in tdtypes.values():
        assert isinstance(dtype, str)


# ---------------------------------------------------------------------
# Binary tensor decoding
# ---------------------------------------------------------------------
def test_binary_tensor_decoding():
    """Check high-rank tensors (>2D) encoded as binary are decoded correctly."""
    n = 3
    data = generate_dummy_data((n, 8, 8, 3), "float")
    features = {"image": data}
    targets = {"label": generate_dummy_data((n, 1), "float")}
    table = build_sample_schema_table(features=features, targets=targets, tags=None)

    coll = SampleCollection(table)
    arr = coll.get_variant_data("features", "image", "raw", fmt=DataFormat.NUMPY)

    assert arr.shape == data.shape
    assert np.allclose(arr, data, atol=1e-6)


# ---------------------------------------------------------------------
# Export and flattening
# ---------------------------------------------------------------------
def test_to_dict_and_to_pandas(sample_collection: SampleCollection):
    """Ensure flattened dict and DataFrame conversions work."""
    dct = sample_collection.to_dict()
    assert isinstance(dct, dict)
    assert any(k.startswith("features.voltage") for k in dct)

    df = sample_collection.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert FEATURES_COLUMN in {c.split(".")[0] for c in df.columns}
    assert df.shape[0] == sample_collection.n_samples


# ---------------------------------------------------------------------
# Roundtrip persistence
# ---------------------------------------------------------------------
def test_parquet_roundtrip(tmp_path, sample_collection: SampleCollection):
    """Confirm Parquet roundtrip retains metadata and schema."""
    path = tmp_path / "collection.parquet"
    pq.write_table(sample_collection.table, path)
    reloaded_tbl = pq.read_table(path)
    coll2 = SampleCollection(reloaded_tbl)

    assert coll2.table_version == SCHEMA_VERSION
    assert coll2.n_samples == sample_collection.n_samples


# ---------------------------------------------------------------------
# Shape + dtype consistency
# ---------------------------------------------------------------------
def test_shape_and_dtype_from_metadata(sample_collection: SampleCollection):
    """Ensure metadata-driven shape/dtype retrieval works."""
    shape = sample_collection.get_variant_shape("features", "voltage", "raw")
    dtype = sample_collection.get_variant_dtype("features", "voltage", "raw")
    assert isinstance(shape, tuple)
    assert isinstance(dtype, str)
    assert "float" in dtype


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
def test_missing_required_domains():
    """Table missing features/targets should raise."""
    bad_table = pa.table({"bad": pa.array([1, 2, 3])})
    with pytest.raises(ValueError, match="Invalid column 'bad'"):
        SampleCollection(bad_table)


# ---------------------------------------------------------------------
# Variant mutation
# ---------------------------------------------------------------------
def test_add_overwrite_variant(sample_collection: SampleCollection):
    # Add transformed feature variant
    scaled_data = sample_collection.get_variant_data(
        domain="features",
        key="voltage",
        variant="raw",
        fmt=DataFormat.NUMPY,
    )
    scaled_data *= 0.5
    sample_collection.add_variant(
        domain="features",
        key="voltage",
        variant="transformed",
        data=scaled_data,
        overwrite=False,
    )

    # Confirm it exists
    existing_variants = sample_collection.get_variant_keys(domain="features", key="voltage")
    assert "transformed" in existing_variants
    assert "raw" in existing_variants

    # Overwrite
    scaled_data *= 0
    sample_collection.add_variant(
        domain="features",
        key="voltage",
        variant="transformed",
        data=scaled_data,
        overwrite=True,
    )
    new_data: np.ndarray = sample_collection.get_variant_data(
        domain="features",
        key="voltage",
        variant="transformed",
        fmt=DataFormat.NUMPY,
    )
    assert np.all(new_data.ravel() == 0)

    # Delete it
    sample_collection.delete_variant(
        domain="features",
        key="voltage",
        variant="transformed",
    )
    existing_variants = sample_collection.get_variant_keys(domain="features", key="voltage")
    assert "transformed" not in existing_variants
    assert "raw" in existing_variants

    # Try deleting "raw"
    with pytest.raises(ValueError, match="cannot be deleted"):
        sample_collection.delete_variant(
            domain="features",
            key="voltage",
            variant="raw",
        )
