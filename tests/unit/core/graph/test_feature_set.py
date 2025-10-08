from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modularml.core.api import FeatureSet, FeatureSubset
from modularml.utils.exceptions import SampleLoadError, SubsetOverlapWarning


# ==========================================================
# Initialization & Basic Properties
# ==========================================================
@pytest.mark.unit
def test_featureset_init(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    assert isinstance(fs, FeatureSet)
    assert fs.label == "NumericFS"
    assert fs.allows_upstream_connections is False
    assert fs.allows_downstream_connections is True
    assert fs.input_shape_spec is None
    assert fs.output_shape_spec is not None
    assert fs.max_inputs == 0
    assert repr(fs).startswith("FeatureSet")


@pytest.mark.unit
def test_clear_subsets(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    fs._subsets["train"] = "dummy"
    fs._split_configs = ["split_cfg"]
    fs.clear_subsets()
    assert len(fs._subsets) == 0
    assert fs._split_configs == []


# ==========================================================
# Subset Management
# ==========================================================
@pytest.mark.unit
def test_add_and_get_subset(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    subset = FeatureSubset(label="train", sample_uuids=[s.uuid for s in fs.samples[:5]], parent=fs)
    fs.add_subset(subset)
    assert fs.get_subset("train") == subset
    assert fs.available_subsets == ["train"]
    assert fs.n_subsets == 1


@pytest.mark.unit
def test_add_duplicate_subset_raises(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    subset = FeatureSubset(label="train", sample_uuids=[s.uuid for s in fs.samples[:5]], parent=fs)
    fs.add_subset(subset)
    with pytest.raises(ValueError, match="already exists"):
        fs.add_subset(subset)


@pytest.mark.unit
def test_add_overlapping_subset_warns(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    s1 = FeatureSubset(label="train", sample_uuids=[s.uuid for s in fs.samples[:5]], parent=fs)
    s2 = FeatureSubset(label="val", sample_uuids=[s.uuid for s in fs.samples[4:10]], parent=fs)
    fs.add_subset(s1)
    with pytest.warns(SubsetOverlapWarning, match="overlapping samples"):
        fs.add_subset(s2)


@pytest.mark.unit
def test_get_invalid_subset_raises(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    with pytest.raises(ValueError, match="not a valid subset"):
        fs.get_subset("missing")


# ==========================================================
# Filtering
# ==========================================================
@pytest.mark.unit
def test_filter_tags(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    tag_key = next(iter(fs.samples[0].tags.keys()))
    tag_val = fs.samples[0].tags[tag_key]
    filtered = fs.filter(**{tag_key: tag_val})
    assert isinstance(filtered, FeatureSet)
    assert len(filtered) >= 1


@pytest.mark.unit
def test_filter_callable(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    feat_key = next(iter(fs.samples[0].features.keys()))
    filtered = fs.filter(**{feat_key: lambda x: np.all(x.value >= 0)})
    assert isinstance(filtered, FeatureSet)


@pytest.mark.unit
def test_filter_nonexistent_key_raises(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    with pytest.raises(ValueError, match="No samples match the provided conditions"):
        _ = fs.filter(nonexistent="x")


# ==========================================================
# Constructors
# ==========================================================
@pytest.mark.unit
def test_from_dict_constructor():
    data = {"x": [1, 2, 3], "y": [10, 20, 30], "tag": ["a", "b", "c"]}
    fs = FeatureSet.from_dict(label="test", data=data, feature_keys="x", target_keys="y", tag_keys="tag")
    assert isinstance(fs, FeatureSet)
    assert len(fs.samples) == 3
    assert list(fs.feature_keys) == ["x"]
    assert list(fs.target_keys) == ["y"]
    assert list(fs.tag_keys) == ["tag"]


@pytest.mark.unit
def test_from_dict_inconsistent_lengths_raises():
    data = {"x": [1, 2, 3], "y": [1, 2]}
    with pytest.raises(ValueError, match="Inconsistent list lengths"):
        FeatureSet.from_dict("bad", data=data, feature_keys="x", target_keys="y")


@pytest.mark.unit
def test_from_pandas_constructor():
    df = pd.DataFrame({"f": [1, 2, 3], "t": [4, 5, 6], "tag": ["a", "b", "c"]})
    fs = FeatureSet.from_pandas(label="pd", df=df, feature_cols="f", target_cols="t", tag_cols="tag")
    assert isinstance(fs, FeatureSet)
    assert len(fs.samples) == 3
    assert list(fs.feature_keys) == ["f"]


# ==========================================================
# Spec Parsing
# ==========================================================
@pytest.mark.unit
def test_parse_spec_valid(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    fs._subsets["train"] = "dummy"  # mock valid subset name
    assert fs._parse_spec("features") == (None, "features", None)
    assert fs._parse_spec("train.features") == ("train", "features", None)
    fkey = fs.feature_keys[0]
    assert fs._parse_spec(f"train.features.{fkey}") == ("train", "features", fkey)


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["badcomponent", "train.invalid", "train.features.badkey"])
def test_parse_spec_invalid(dummy_featureset_numeric, invalid):
    fs = dummy_featureset_numeric
    fs._subsets["train"] = "dummy"
    with pytest.raises(ValueError, match="Invalid"):
        fs._parse_spec(invalid)


# ==========================================================
# Save / Load Samples
# ==========================================================
@pytest.mark.unit
def test_save_and_load_samples(tmp_path, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    file_path = tmp_path / "samples"
    fs.save_samples(file_path)
    loaded = FeatureSet.load_samples(file_path)
    assert isinstance(loaded, list)
    assert isinstance(loaded[0], type(fs.samples[0]))


# ==========================================================
# Config Serialization
# ==========================================================
@pytest.mark.unit
def test_get_and_from_config(tmp_path, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    file_path = tmp_path / "samples"
    fs.save_samples(file_path)
    cfg = fs.get_config(sample_path=file_path)
    fs2 = FeatureSet.from_config(cfg)
    assert isinstance(fs2, FeatureSet)
    assert len(fs2.samples) == len(fs.samples)


@pytest.mark.unit
def test_from_config_missing_path_raises():
    cfg = {"label": "x", "sample_data": None}
    with pytest.raises(ValueError, match="missing 'sample_data'"):
        FeatureSet.from_config(cfg)


@pytest.mark.unit
def test_from_config_load_failure(tmp_path):
    bad_path = tmp_path / "missing"
    cfg = {"label": "x", "sample_data": str(bad_path)}
    with pytest.raises(SampleLoadError):
        FeatureSet.from_config(cfg)


@pytest.mark.unit
def test_to_serializable_and_back(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    obj = fs.to_serializable()
    fs2 = FeatureSet.from_serializable(obj)
    assert isinstance(fs2, FeatureSet)
    assert fs2.label == fs.label
    assert len(fs2.samples) == len(fs.samples)


# ==========================================================
# Joblib Save / Load
# ==========================================================
@pytest.mark.unit
def test_save_and_load_joblib(tmp_path, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    path = tmp_path / "fs.joblib"
    fs.save(path)
    assert Path(path).exists()
    loaded = FeatureSet.load(path)
    assert isinstance(loaded, FeatureSet)
    assert loaded.label == fs.label


@pytest.mark.unit
def test_save_overwrite_and_fileexists(tmp_path, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    path = tmp_path / "fs.joblib"
    fs.save(path)
    with pytest.raises(FileExistsError):
        fs.save(path, overwrite_existing=False)
    fs.save(path, overwrite_existing=True)
    assert Path(path).exists()


@pytest.mark.unit
def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FeatureSet.load(tmp_path / "doesnotexist")


# ==========================================================
# Transform Record Lifecycle
# ==========================================================
@pytest.mark.unit
def test_transform_record_serialization(tmp_path):
    from modularml.core.graph.feature_set import TransformRecord
    from modularml.core.transforms.feature_transform import FeatureTransform

    # Create dummy transform
    ft = FeatureTransform(scaler="minmax")
    record = TransformRecord(fit_spec="features", apply_spec="features", transform=ft)

    obj = record.to_serializable()
    record2 = TransformRecord.from_serializable(obj)
    assert record2.fit_spec == "features"
    assert isinstance(record2.transform, FeatureTransform)

    path = tmp_path / "transform"
    record.save(path)
    loaded = TransformRecord.load(path)
    assert isinstance(loaded, TransformRecord)
    with pytest.raises(FileExistsError):
        record.save(path, overwrite_existing=False)
