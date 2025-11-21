import numpy as np
import pandas as pd
import pytest

from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.graph.featureset import FeatureSet
from modularml.core.references.featureset_ref import FeatureSetRef
from modularml.core.splitting.condition_splitter import ConditionSplitter
from modularml.core.splitting.random_splitter import RandomSplitter
from modularml.utils.exceptions import SplitOverlapWarning


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


# ==========================================================
# Constructors
# ==========================================================
@pytest.mark.unit
def test_from_dict_constructor():
    data = {"x": [1, 2, 3], "y": [10, 20, 30], "tag": ["a", "b", "c"]}
    fs = FeatureSet.from_dict(label="test", data=data, feature_keys="x", target_keys="y", tag_keys="tag")
    assert isinstance(fs, FeatureSet)
    sc = fs.collection
    assert sc.n_samples == 3
    assert sc.feature_keys == ["x"]
    assert sc.target_keys == ["y"]
    assert sc.tag_keys == ["tag"]


@pytest.mark.unit
def test_from_dict_inconsistent_lengths_raises():
    data = {"x": [1, 2, 3], "y": [1, 2]}
    with pytest.raises(ValueError, match="expected length 3 but got length 2"):
        FeatureSet.from_dict("bad", data=data, feature_keys="x", target_keys="y")


@pytest.mark.unit
def test_from_pandas_constructor():
    df = pd.DataFrame({"f": [1, 2, 3], "t": [4, 5, 6], "tag": ["a", "b", "c"]})
    fs = FeatureSet.from_pandas(label="pd", df=df, feature_cols="f", target_cols="t", tag_cols="tag")
    sc = fs.collection
    assert sc.n_samples == 3
    assert sc.feature_keys == ["f"]


# ==========================================================
# Filtering
# ==========================================================
@pytest.mark.unit
def test_filter_tags(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    sc = fs.collection
    tag_key = sc.tag_keys[0]
    tag_val = sc.get_tags()[tag_key][0]

    fsv = fs.filter(**{tag_key: tag_val})
    assert isinstance(fsv, FeatureSetView)
    assert fsv.source is fs
    assert len(fsv) >= 1


@pytest.mark.unit
def test_filter_callable(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    sc = fs.collection
    f_key = sc.feature_keys[0]

    fsv = fs.filter(**{f_key: lambda x: np.all(x >= 0)})
    assert isinstance(fsv, FeatureSetView)


@pytest.mark.unit
def test_filter_nonexistent_key_raises(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    with pytest.raises(KeyError):
        fs.filter(nonexistent="xyz")


# ==========================================================
# Splitting
# ==========================================================
@pytest.mark.unit
def test_random_split_basic(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    all_ids = set(fs.collection.to_pandas()["sample_id"])

    splitter = RandomSplitter({"train": 0.8, "test": 0.2})
    views = splitter.split(fs._as_view(), return_views=True)
    df_train = views["train"].to_pandas()
    df_test = views["test"].to_pandas()

    train_ids = set(df_train["sample_id"].values)
    test_ids = set(df_test["sample_id"].values)

    # Validate disjoint and coverage
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.issubset(all_ids)
    assert test_ids.issubset(all_ids)

    # Validate approximate ratio
    ratio = len(df_train) / len(fs)
    assert 0.75 < ratio < 0.85


@pytest.mark.unit
def test_random_split_group_by(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    all_ids = set(fs.collection.to_pandas()["sample_id"].values)

    splitter = RandomSplitter({"train": 0.8, "test": 0.2}, group_by="T_STR")
    views = splitter.split(fs._as_view(), return_views=True)
    df_train = views["train"].to_pandas()
    df_test = views["test"].to_pandas()

    train_ids = set(df_train["sample_id"].values)
    test_ids = set(df_test["sample_id"].values)

    # No overlap
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.issubset(all_ids)
    assert test_ids.issubset(all_ids)

    # Check that T_STR groups are disjoint between splits
    assert set(df_train["tags.T_STR.raw"].unique()).isdisjoint(df_test["tags.T_STR.raw"].unique())


@pytest.mark.unit
def test_condition_split(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    all_ids = set(fs.collection.to_pandas()["sample_id"].values)

    splitter = ConditionSplitter(
        train={"T_STR": ["red"]},
        test={"T_STR": ["blue"]},
    )
    views = splitter.split(fs._as_view(), return_views=True)

    df_train = views["train"].to_pandas()
    df_test = views["test"].to_pandas()

    train_ids = set(df_train["sample_id"].values)
    test_ids = set(df_test["sample_id"].values)

    # Ensure disjoint and coverage
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.issubset(all_ids)
    assert test_ids.issubset(all_ids)

    # Check correct filtering
    assert np.all(df_train["tags.T_STR.raw"].unique() == ["red"])
    assert np.all(df_test["tags.T_STR.raw"].unique() == ["blue"])


@pytest.mark.unit
def test_builtin_split_methods(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    fs.split_random(
        ratios={"train": 0.5, "val": 0.2, "test": 0.3},
        register=True,
        return_views=False,
    )

    assert fs.available_splits == ["train", "val", "test"]

    fsv_train = fs.get_split(split_name="train")
    frac = len(fsv_train) / fs.collection.n_samples
    assert abs(frac - 0.5) <= 0.05

    # Alias access
    assert fsv_train is fs.get_split("train")

    # Splitting a view (will overlap)
    with pytest.warns(SplitOverlapWarning):
        fsv_train.split_random(
            ratios={"train_p1": 0.5, "train_p2": 0.5},
            register=True,
            return_views=False,
        )

    assert set(fs.available_splits) == {"train", "val", "test", "train_p1", "train_p2"}

    fsv_train_p1 = fs.get_split("train_p1")
    frac2 = len(fsv_train_p1) / len(fsv_train)
    assert abs(frac2 - 0.5) <= 0.05

    # Clear splits
    fs.clear_splits()
    assert fs.available_splits == []


# ==========================================================
# Split configuration tracking
# ==========================================================
@pytest.mark.unit
def test_split_config_tracking_and_order(dummy_featureset_numeric):
    """Verify that FeatureSet._split_configs correctly reflects the order of applied splits."""
    fs = dummy_featureset_numeric

    # Apply multiple splits in a defined order
    fs.split_random(ratios={"train": 0.6, "val": 0.4}, register=True)
    with pytest.warns(SplitOverlapWarning):
        fs.split_by_condition(
            conditions={
                "group_red_green": {"T_STR": ["red", "green"]},
                "group_blue": {"T_STR": ["blue"]},
            },
            register=True,
        )

    # Ensure both splitters are recorded
    assert len(fs._split_configs) == 2

    # Validate order: random splitter was applied first
    first_cfg, second_cfg = fs._split_configs
    assert "ratios" in first_cfg.splitter_state
    assert "conditions" in second_cfg.splitter_state

    assert all(isinstance(r.applied_to, FeatureSetRef) for r in fs._split_configs)

    # Clear splits and confirm configs are also cleared
    fs.clear_splits()
    assert fs.available_splits == []
    assert fs._split_configs == []
