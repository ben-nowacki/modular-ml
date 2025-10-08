import pytest

from modularml.core.data_structures.data import Data
from modularml.core.splitters.cross_validation import CrossValidationSplitter


# ==========================================================
# Basic K-fold splitting (no grouping)
# ==========================================================
@pytest.mark.unit
def test_cross_validation_basic(dummy_featureset_numeric):
    """Test basic K-fold split correctness and disjointness."""
    splitter = CrossValidationSplitter(n_folds=5, seed=123)
    folds = splitter.split(dummy_featureset_numeric.samples)

    # Expect 2 * n_folds splits (train + val)
    assert len(folds) == 2 * splitter.n_folds
    for f in range(splitter.n_folds):
        assert f"train.fold_{f}" in folds
        assert f"val.fold_{f}" in folds

    # Ensure each fold covers all samples, disjoint train/val
    all_ids = {s.uuid for s in dummy_featureset_numeric.samples}
    for f in range(splitter.n_folds):
        train_ids = set(folds[f"train.fold_{f}"])
        val_ids = set(folds[f"val.fold_{f}"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.union(val_ids) == all_ids


# ==========================================================
# Grouped K-fold splitting
# ==========================================================
@pytest.mark.unit
def test_cross_validation_grouped(dummy_featureset_numeric):
    """Ensure group-based splitting keeps groups intact."""
    # Add cell_id tag grouping to simulate multiple groups
    for i, s in enumerate(dummy_featureset_numeric.samples):
        s.tags["cell_id"] = Data(i // 50)  # 20 groups total

    splitter = CrossValidationSplitter(n_folds=5, group_by="cell_id", seed=42)
    folds = splitter.split(dummy_featureset_numeric.samples)

    # Confirm expected folds exist
    assert all(f"{lbl}.fold_{f}" in folds for lbl in ("train", "val") for f in range(5))

    # Collect all group assignments
    group_assignments = {}
    for f in range(5):
        val_ids = set(folds[f"val.fold_{f}"])
        for s in dummy_featureset_numeric.samples:
            group = s.tags["cell_id"]
            if s.uuid in val_ids:
                group_assignments.setdefault(group, []).append(f)

    # Each group should appear in *exactly one* validation fold
    val_counts = [len(set(v)) for v in group_assignments.values()]
    assert all(c == 1 for c in val_counts)


# ==========================================================
# Reproducibility
# ==========================================================
@pytest.mark.unit
def test_cross_validation_reproducibility(dummy_featureset_numeric):
    """Cross-validation should be reproducible given the same seed."""
    splitter_a = CrossValidationSplitter(n_folds=4, seed=123)
    splitter_b = CrossValidationSplitter(n_folds=4, seed=123)
    folds_a = splitter_a.split(dummy_featureset_numeric.samples)
    folds_b = splitter_b.split(dummy_featureset_numeric.samples)

    # Order and membership should be identical
    assert folds_a.keys() == folds_b.keys()
    for key in folds_a:
        assert folds_a[key] == folds_b[key]


@pytest.mark.unit
def test_cross_validation_different_seed_changes_split(dummy_featureset_numeric):
    """Changing the seed should yield different splits."""
    splitter_a = CrossValidationSplitter(n_folds=4, seed=123)
    splitter_b = CrossValidationSplitter(n_folds=4, seed=456)
    folds_a = splitter_a.split(dummy_featureset_numeric.samples)
    folds_b = splitter_b.split(dummy_featureset_numeric.samples)

    # Some folds should differ
    diffs = [folds_a[k] != folds_b[k] for k in folds_a]
    assert any(diffs)


# ==========================================================
# Config serialization
# ==========================================================
@pytest.mark.unit
def test_cross_validation_config_roundtrip():
    """Test that get_config() and from_config() are symmetric."""
    splitter = CrossValidationSplitter(n_folds=3, group_by="cell_id", seed=99)
    cfg = splitter.get_config()
    new_splitter = CrossValidationSplitter.from_config(cfg)

    assert new_splitter.n_folds == splitter.n_folds
    assert new_splitter.group_by == splitter.group_by
    assert new_splitter.seed == splitter.seed
    assert new_splitter.train_lbl == splitter.train_lbl
    assert new_splitter.val_lbl == splitter.val_lbl


# ==========================================================
# Edge cases
# ==========================================================
@pytest.mark.unit
def test_cross_validation_invalid_fold_count():
    """Ensure invalid fold count raises ValueError."""
    with pytest.raises(ValueError, match="n_folds must be >= 2 for cross-validation"):
        CrossValidationSplitter(n_folds=1)
