import pytest

from modularml.core.api import FeatureSet, FeatureSubset


# ==========================================================
# Initialization & Basic Properties
# ==========================================================
def test_featuresubset_init(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    uuids = [s.uuid for s in fs.samples[:10]]
    subset = FeatureSubset(label="train", parent=fs, sample_uuids=uuids)

    assert isinstance(subset, FeatureSubset)
    assert subset.label == "train"
    assert subset.parent == fs
    assert len(subset) == len(uuids)
    assert subset.output_shape_spec is not None
    assert subset.max_inputs == 1
    assert subset.allows_upstream_connections
    assert subset.allows_downstream_connections
    assert repr(subset).startswith("FeatureSubset(")


def test_featuresubset_invalid_uuids_raise(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    bad_uuids = ["not-a-real-uuid"]
    with pytest.raises(ValueError, match="Invalid sample_uuids"):
        FeatureSubset(label="bad", parent=fs, sample_uuids=bad_uuids)


def test_featuresubset_dead_parent_raises(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    uuids = [s.uuid for s in fs.samples[:3]]
    subset = FeatureSubset(label="orphan", parent=fs, sample_uuids=uuids)

    ## Manually simulate dead weakref
    subset._parent_ref = lambda: None
    with pytest.raises(ReferenceError, match="no longer exists"):
        _ = subset.parent


# ==========================================================
# Disjointness Checking
# ==========================================================
def test_is_disjoint_with(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    s1 = FeatureSubset(label="A", parent=fs, sample_uuids=[s.uuid for s in fs.samples[:5]])
    s2 = FeatureSubset(label="B", parent=fs, sample_uuids=[s.uuid for s in fs.samples[5:10]])
    s3 = FeatureSubset(label="C", parent=fs, sample_uuids=[s.uuid for s in fs.samples[3:8]])

    assert s1.is_disjoint_with(s2) is True
    assert s1.is_disjoint_with(s3) is False


# ==========================================================
# Split & Random Split
# ==========================================================
def test_split_method_calls_splitter(monkeypatch, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    uuids = [s.uuid for s in fs.samples[:20]]
    subset = FeatureSubset(label="train", parent=fs, sample_uuids=uuids)

    class DummySplitter:
        def split(self, samples):
            # Return dict of label->uuid list
            return {"fold1": [s.uuid for s in samples[:10]], "fold2": [s.uuid for s in samples[10:]]}

    splitter = DummySplitter()

    # Mock parent add_subset and _add_split_config
    fs.add_subset = lambda s: setattr(fs, f"added_{s.label}", True)
    fs._add_split_config = lambda **kwargs: setattr(fs, "split_config_added", True)  # noqa: ARG005

    new_subsets = subset.split(splitter)
    assert all(isinstance(s, FeatureSubset) for s in new_subsets)
    assert hasattr(fs, "split_config_added")
    assert hasattr(fs, "added_fold1")
    assert hasattr(fs, "added_fold2")


def test_split_random_creates_subsets(monkeypatch, dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    uuids = [s.uuid for s in fs.samples[: len(fs) // 4]]
    subset = FeatureSubset(label="train", parent=fs, sample_uuids=uuids)

    called = {}

    class DummyRandomSplitter:
        def __init__(self, ratios, seed):
            called["ratios"] = ratios
            called["seed"] = seed

        def split(self, samples):
            n = len(samples)
            mid = n // 2
            return {"half1": [s.uuid for s in samples[:mid]], "half2": [s.uuid for s in samples[mid:]]}

        def get_config(self):
            return {}

    import importlib

    rs = importlib.import_module("modularml.core.splitters.random_splitter")
    monkeypatch.setattr(rs, "RandomSplitter", DummyRandomSplitter)

    new_subsets = subset.split_random(ratios={"half1": 0.5, "half2": 0.5}, seed=99)
    assert len(new_subsets) == 2
    assert called["ratios"] == {"half1": 0.5, "half2": 0.5}
    assert called["seed"] == 99


# ==========================================================
# Config Serialization
# ==========================================================
def test_get_config_and_from_config(dummy_featureset_numeric):
    fs = dummy_featureset_numeric
    uuids = [s.uuid for s in fs.samples[:5]]
    subset = FeatureSubset(label="train", parent=fs, sample_uuids=uuids)
    cfg = subset.get_config()

    assert cfg["label"] == "train"
    assert cfg["parent_label"] == fs.label
    assert set(cfg["sample_uuids"]) == set(uuids)

    subset2 = FeatureSubset.from_config(cfg, parent=fs)
    assert isinstance(subset2, FeatureSubset)
    assert subset2.label == subset.label
    assert subset2.parent == fs


def test_from_config_wrong_parent_raises(dummy_featureset_numeric):
    fs1 = dummy_featureset_numeric
    fs2 = FeatureSet.from_dict(label="Other", data={"x": [1, 2, 3], "y": [4, 5, 6]}, feature_keys="x", target_keys="y")
    uuids = [s.uuid for s in fs1.samples[:5]]
    subset = FeatureSubset(label="train", parent=fs1, sample_uuids=uuids)
    cfg = subset.get_config()

    with pytest.raises(ValueError, match="does not match the config definition"):
        FeatureSubset.from_config(cfg, parent=fs2)
