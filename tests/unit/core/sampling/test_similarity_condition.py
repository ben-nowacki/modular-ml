import math

from modularml.core.sampling.similiarity_condition import SimilarityCondition


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def approx_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------
# Basic "similar" mode tests
# ---------------------------------------------------------
def test_binary_similar_match():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.5,
        allow_fallback=False,  # if False, custom weights are ignored
        weight_mode="binary",
        min_weight=0.1,
        max_weight=100,
    )
    assert cond.score(1.0, 1.2) == 1

    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.5,
        allow_fallback=True,
        weight_mode="binary",
        min_weight=0.1,
        max_weight=100,
    )
    assert cond.score(1.0, 1.2) == 100  # diff=0.2 <= tol


def test_binary_similar_nonmatch_no_fallback():
    # Non-matach + no fallback -> 0
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.1,
        allow_fallback=False,
        weight_mode="binary",
    )
    assert cond.score(1.0, 2.0) == 0.0  # diff=1.0 > tol → no fallback → 0.0


def test_binary_similar_nonmatch_with_fallback():
    # Non-match + binary -> min_weight
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.1,
        allow_fallback=True,
        weight_mode="binary",
        min_weight=0.05,
    )
    assert cond.score(1.0, 2.0) == 0.05

    # If Fallback if False, should override "weight_mode"
    # Returns 0 if non-match, 1 if match
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.1,
        allow_fallback=False,
        weight_mode="binary",
        min_weight=0.05,
    )
    assert cond.score(1.0, 2.0) == 0


def test_tolerance_boundary():
    # Match + binary -> max weight
    cond = SimilarityCondition(
        mode="similar",
        tolerance=1.0,
        allow_fallback=True,
        weight_mode="binary",
        max_weight=1,
    )
    assert cond.score(1.0, 2.0) == 1.0


# ---------------------------------------------------------
# Dissimilar mode tests
# ---------------------------------------------------------
def test_dissimilar_mode_match():
    # Diff > tol = match since dissimilar
    # Match + binary -> max_weight
    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=0.5,
        weight_mode="binary",
        allow_fallback=True,
        max_weight=1.0,
    )
    assert cond.score(1.0, 2.0) == 1.0


def test_dissimilar_mode_nonmatch_no_fallback():
    # Dissim. non-match + no-fallback -> 0
    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=2.0,
        allow_fallback=False,
    )
    assert cond.score(1.0, 1.5) == 0.0


def test_dissimilar_mode_nonmatch_with_fallback():
    # Dissim. non-match (diff < tol) + fallback -> min_weight
    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=2.0,
        allow_fallback=True,
        weight_mode="binary",
        min_weight=0.2,
    )
    assert cond.score(1.0, 1.5) == 0.2


# ---------------------------------------------------------
# Linear weighting tests
# ---------------------------------------------------------
def test_linear_similar_high_weight():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.5,
        allow_fallback=True,
        weight_mode="linear",
        min_weight=0.1,
        max_weight=100,
    )
    # diff = 0.1 -> weight = 0.5 / 0.1 = 5.0
    assert approx_equal(cond.score(1.0, 1.1), 5.0)


def test_linear_similar_min_clip():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.5,
        allow_fallback=True,
        weight_mode="linear",
        min_weight=0.2,
    )
    # diff=10 -> 0.5/10 = 0.05 -> clipped to min_weight=0.2
    assert approx_equal(cond.score(1.0, 11.0), 0.2)


def test_linear_similar_max_clip():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=1.0,
        allow_fallback=True,
        weight_mode="linear",
        max_weight=10,
    )
    # diff=0.00001 -> weight=100000 -> clipped to 10
    assert approx_equal(cond.score(1.0, 1.00001), 10.0)


def test_linear_dissimilar():
    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=1.0,
        allow_fallback=True,
        weight_mode="linear",
        min_weight=0.1,
        max_weight=100,
    )
    # dissimilar mode -> weight = diff/tol -> diff=5 -> 5/1 = 5
    assert approx_equal(cond.score(1.0, 6.0), 5.0)


# ---------------------------------------------------------
# Exponential weighting tests
# ---------------------------------------------------------
def test_exp_similar():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.5,
        allow_fallback=True,
        weight_mode="exp",
        min_weight=0.1,
        max_weight=100,
    )
    # diff=0.2 -> exp(1 - diff/tol) = exp(1 - 0.4) = exp(0.6)
    assert approx_equal(cond.score(1.0, 1.2), math.exp(0.6))


def test_exp_dissimilar():
    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=0.5,
        allow_fallback=True,
        weight_mode="exp",
        min_weight=0.1,
        max_weight=100,
    )
    # diff=1.5 -> exp(diff/tol - 1) = exp(3 - 1) = exp(2)
    assert approx_equal(cond.score(1.0, 2.5), math.exp(2))


def test_exp_max_clip():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=1.0,
        allow_fallback=True,
        weight_mode="exp",
        min_weight=0.1,
        max_weight=2.0,
    )
    assert cond.score(1.0, 1.00001) <= 2.0


# ---------------------------------------------------------
# Custom metric tests
# ---------------------------------------------------------
def test_custom_metric():
    # metric is used to determine the difference between
    # two values
    # If resulting difference is <= toleranace -> is match
    # Otherwise is not a match
    # Custom metrics are not used for weighting
    cond = SimilarityCondition(
        tolerance=1.0,
        metric=lambda a, b: (a - b) ** 2,  # squared distance
        weight_mode="linear",
        min_weight=0.1,
        max_weight=100,
    )
    diff = (2.0 - 3.0) ** 2  # (custom metric) diff = 1
    expected = 1.0 / diff  # (linear weighting) tol / diff = 1.0
    assert approx_equal(cond.score(2.0, 3.0), expected)


# ---------------------------------------------------------
# Non-numeric equality behavior
# ---------------------------------------------------------
def test_non_numeric_equal():
    # Non-numeric values are compared for equality
    # if a == b, diff = 0, else diff = 1
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.0,
        allow_fallback=False,
    )
    assert cond.score("abc", "abc") == 1.0


def test_non_numeric_not_equal_no_fallback():
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.0,
        allow_fallback=False,
    )
    assert cond.score("abc", "xyz") == 0.0


def test_non_numeric_not_equal_with_fallback():
    # Non-numeric always have 0 or 1 differences
    # We can change the weight values using "binary" mode
    # Other modes are accepted but will only ever give
    # you two values since the difference is only ever 0 or 1
    cond = SimilarityCondition(
        mode="similar",
        tolerance=0.0,
        allow_fallback=True,
        weight_mode="binary",
        min_weight=0.3,
        max_weight=100,
    )
    assert cond.score("abc", "xyz") == 0.3

    cond = SimilarityCondition(
        mode="dissimilar",
        tolerance=0.0,
        allow_fallback=True,
        weight_mode="binary",
        min_weight=0.3,
        max_weight=100,
    )
    assert cond.score("abc", "xyz") == 100
