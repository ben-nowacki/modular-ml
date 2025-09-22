from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class SimilarityCondition:
    """
    A flexible rule for defining similarity or dissimilarity between two samples.

    Attributes:
        field ({"tags","features","targets"}):
            Which part of the sample to compare.
              - "tags": Compares values in the sample's tags dictionary.
              - "features": Compares feature values.
              - "targets": Compares target labels/values.
        key (str | None, default=None):
            Name of the attribute to compare within the chosen field.
            For "tags", this is the tag key (e.g., "cell_id", "SOH_PCT").
            For "features" or "targets", this is the feature/target name.
            If None, the entire field may be compared (depending on sampler logic).
        mode ({"similar","dissimilar"}, default="similar"):
            Whether to select pairs that are within tolerance ("similar")
            or explicitly outside tolerance ("dissimilar").
        tolerance (float, default=0.0):
            Numeric threshold for values to be considered "similar".
            Example: tolerance=0.05 means |a - b| <= 0.05 is a valid match.
        metric (Callable[[float, float], float] | None, default=None):
            Optional custom distance function. If not provided:
              - Numeric types use absolute difference.
              - Other types use equality.
        weight_mode ({"uniform","linear","exp"}, default="uniform"):
            Strategy for assigning weights to valid or fallback pairs:
              - "uniform": All matches get weight 1.0; non-matches get 0.1 if
                `allow_fallback=True`.
              - "linear": Weight = tolerance / diff (≥1 for matches, <1 for non-matches).
              - "exp": Weight = exp(1 - diff / tolerance), clipped at `max_weight`.
        max_weight (float, default=100):
            Maximum weight allowed when using `weight_mode != 'uniform'`
        min_weight (float, default=0.1):
            Minimum weight allowed when using `weight_mode != 'uniform'`
        allow_fallback (bool, default=False):
            If True, pairs that fail the match condition are not discarded;
            they are instead given a down-weighted score (<1).
            If False, non-matches always score 0.0.

    Weight semantics:
        - Matches always receive weight ≥ 1.0 (better matches → larger weight).
        - Non-matches receive weight < 1.0 if `allow_fallback=True`, else 0.0.

    Examples:
        >>> cond = SimilarityCondition(field="tags", key="SOH_PCT", mode="similar", tolerance=0.5, weight_mode="linear")
        >>> cond.score(0.80, 0.82)
        2.5   # diff=0.02, tol=0.5 -> high weight
        >>> cond.score(0.80, 0.90)
        1.0   # diff=0.10, tol=0.5 -> valid but weaker
        >>> cond.score(0.80, 1.5)
        0.33  # diff=0.70 > tol=0.5, fallback weight

    """

    field: Literal["tags", "features", "targets"]
    key: str
    mode: Literal["similar", "dissimilar"] = "similar"

    tolerance: float = 0.0
    metric: Callable[[float, float], float] | None = None

    weight_mode: Literal["uniform", "linear", "exp"] = "uniform"
    max_weight: float = 100
    min_weight: float = 0.1
    allow_fallback: bool = False

    def score(self, a, b) -> float:  # noqa: PLR0911
        """
        Compute the similarity/dissimilarity score between two values.

        Args:
            a: First value (anchor).
            b: Second value (candidate).

        Returns:
            float: A non-negative weight score.

        Notes:
            - For matches (diff <= tolerance if mode="similar"):
                Score >= 1.0 (better matches = larger weight).
            - For non-matches:
                Score < 1.0 if `allow_fallback=True`.
                Score = 0.0 if `allow_fallback=False`.

        """
        # Step 1: compute difference
        if self.metric:
            diff = abs(self.metric(a, b))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = abs(a - b)
        else:
            diff = 0 if a == b else float("inf")

        # Step 2: check match condition
        is_match = diff <= self.tolerance
        if self.mode == "dissimilar":
            is_match = not is_match
        if not is_match and not self.allow_fallback:
            return 0.0

        # Step 3: assign weight
        # - All valid matches get a weight >= 1
        # - All invalid matches get a weight < 1
        if self.weight_mode == "uniform":
            return 1.0 if is_match else self.min_weight

        if self.weight_mode == "linear":
            # Inverse linear weight based on how much better than match condition this pairing is
            if self.mode == "similar":
                # Smaller diff = higher weight
                # e.g. diff=0.2, tol=0.5 -> 2.5
                return np.clip(
                    self.tolerance / max(diff, 1e-9),
                    a_max=self.max_weight,
                    a_min=self.min_weight,
                ).__float__()
            # Dissimilar = Larger diff -> higher weight
            # e.g. diff=1.0, tol=0.5 -> 2.0
            return np.clip(
                diff / max(self.tolerance, 1e-9),
                a_max=self.max_weight,
                a_min=self.min_weight,
            ).__float__()

        if self.weight_mode == "exp":
            # Exponential weight based on how much better than match condition this pairing is
            if self.mode == "similar":
                # Smaller diff = higher weight
                # e.g. diff=0.2, tol=0.5 -> np.exp(1 - 0.2/0.5) = 1.822
                return np.clip(
                    float(np.exp(1 - diff / max(self.tolerance, 1e-9))),
                    a_max=self.max_weight,
                    a_min=self.min_weight,
                ).__float__()
            # Dissimilar = Larger diff -> higher weight
            # e.g., diff=1.0, tol=0.5 -> np.exp(1/0.5 - 1) = 2.718
            return np.clip(
                float(np.exp(diff / max(self.tolerance, 1e-9) - 1)),
                a_max=self.max_weight,
                a_min=self.min_weight,
            ).__float__()

        return 0.0


@dataclass
class ConditionBucket:
    """Container for all samples that fall into a specific condition bucket."""

    key: Any  # The "bucket key" (e.g. tag value, bin, etc.)
    matching_samples: set[str] = field(default_factory=set)
    fallback_samples: set[str] = field(default_factory=set)

    def add_match(self, sample_uuid: str):
        """Add a sample UUID to the matching set."""
        self.matching_samples.add(sample_uuid)

    def add_fallback(self, sample_uuid: str):
        """Add a sample UUID to the fallback set."""
        self.fallback_samples.add(sample_uuid)

    def all_samples(self) -> set[str]:
        """Return all samples in this bucket (matches + fallback)."""
        return self.matching_samples | self.fallback_samples
