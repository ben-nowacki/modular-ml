import warnings
from itertools import product
from typing import Any

import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.feature_subset import FeatureSubset
from modularml.core.samplers.condition import ConditionBucket, SimilarityCondition
from modularml.core.samplers.feature_sampler import FeatureSampler
from modularml.utils.data_format import to_python


class PairedSampler(FeatureSampler):
    """
    A sampler that constructs (anchor, pair) batches based on user-defined similarity conditions.

    This class generates sample pairs from a `FeatureSet` or `FeatureSubset` using one or more
    `SimilarityCondition`s. Each sample is treated as an "anchor", and candidate samples are
    drawn from condition-specific "matching" or "fallback" pools. Pairs can be exhaustive or
    limited in number per anchor.

    Attributes:
        conditions (list[SimilarityCondition]):
            List of conditions defining how samples should be matched or dissimilar.
        max_pairs_per_anchor (int | None):
            Maximum number of pairs to generate for each anchor sample.
            - If None: all valid pairs are generated (can grow quadratically).
            - If int: at most N pairs per anchor are generated, preferring matches over fallbacks.
        _cached_sample_bucket_keys (dict[str, tuple]):
            Internal cache mapping sample UUIDs → their bucket keys for faster lookup.

    """

    def __init__(
        self,
        conditions: list[SimilarityCondition],
        source: FeatureSet | FeatureSubset | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        max_pairs_per_anchor: int | None = 3,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a PairedSampler.

        Args:
            conditions (list[SimilarityCondition]):
                The conditions that govern how sample pairs are formed and weighted.
            source (FeatureSet | FeatureSubset, optional):
                Data source to draw samples from. If None, you must call `bind_source()` later.
            batch_size (int, default=1):
                Number of pairs to include per batch.
            shuffle (bool, default=False):
                Whether to shuffle pairs within batches and/or the batch order.
            max_pairs_per_anchor (int | None, default=3):
                Maximum number of pairs per anchor. If None, all valid pairs are generated.
            drop_last (bool, default=False):
                Whether to drop the final batch if it is smaller than `batch_size`.
            seed (int, optional):
                Random seed for reproducibility.

        """
        self.conditions = conditions
        self.max_pairs_per_anchor = max_pairs_per_anchor
        # Cache for faster lookup during pair generation
        self._cached_sample_bucket_keys: dict[str, tuple] = {}
        super().__init__(
            source=source,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def _get_sample_cond_values(self, sample: Sample) -> list[Any]:
        """
        Extract the raw values from a sample for each condition.

        Args:
            sample (Sample): The sample from which to extract condition values.

        Returns:
            list[Any]: Values corresponding to each condition in `self.conditions`.

        Notes:
            - For "tags", the sample's tag dict is queried with the condition key.
            - For "features" or "targets", the sample's respective dict is queried.
            - Scalars wrapped in single-element lists/tuples are unwrapped.

        """
        vals = []
        for cond in self.conditions:
            v = to_python(sample.__getattribute__(cond.field)[cond.key])
            if isinstance(v, (tuple, list)) and len(v) == 1:
                v = v[0]
            vals.append(v)
        return vals

    def _get_sample_bucket_keys(self, sample: Sample) -> tuple[Any]:
        """
        Compute (and cache) the bucket keys for a given sample across all conditions.

        Args:
            sample (Sample): The sample whose bucket keys should be computed.

        Returns:
            tuple[Any]: A tuple of bucket keys, one per condition.

        Raises:
            ValueError: If the condition key is missing from the sample.

        Notes:
            - Numeric values are bucketed using `floor(value / tolerance)` if tolerance > 0.
            - Non-numeric values are used directly as bucket keys.
            - Results are cached by sample UUID for speed.

        """
        if sample.uuid not in self._cached_sample_bucket_keys:
            b_keys = []
            try:
                for cond in self.conditions:
                    val = to_python(sample.__getattribute__(cond.field)[cond.key])
                    if isinstance(val, (tuple, list)) and len(val) == 1:
                        val = val[0]

                    if isinstance(val, (float, int)):
                        bucket_key = int(val // cond.tolerance) if cond.tolerance > 0 else val
                    else:
                        bucket_key = val

                    b_keys.append(bucket_key)

            except AttributeError as e:
                msg = f"SimilarityCondition key does not exist in sample: {cond.field}.{cond.key}. {e}"
                raise ValueError(msg) from e

            self._cached_sample_bucket_keys[sample.uuid] = tuple(b_keys)

        return self._cached_sample_bucket_keys[sample.uuid]

    def _build_condition_buckets(self) -> list[dict[Any, ConditionBucket]]:
        """
        Build condition-specific buckets for all samples.

        Returns:
            list[dict[Any, ConditionBucket]]:
                A list of dicts (one per condition). Each dict maps bucket keys
                → `ConditionBucket` containing matching/fallback samples.

        Notes:
            - "Similar" mode → samples with identical keys go into matching sets.
            - "Dissimilar" mode → samples with different keys go into matching sets.
            - Fallback samples are added when `allow_fallback=True`.

        """
        n_conditions = len(self.conditions)

        # Initiallize all buckets keys
        all_buckets: list[dict[Any, ConditionBucket]] = [{} for _ in range(n_conditions)]
        for s in self.source.samples:
            b_keys = self._get_sample_bucket_keys(sample=s)
            for i, key in enumerate(b_keys):
                if key not in all_buckets[i]:
                    all_buckets[i][key] = ConditionBucket(key=key)

        # Build up buckets with matching & fallback samples
        for s in self.source.samples:
            sample_bucket_keys = self._get_sample_bucket_keys(sample=s)
            zipped = zip(self.conditions, sample_bucket_keys, strict=True)
            for i, (cond, sbk) in enumerate(zipped):
                # Go through all buckets
                for bk, bucket in all_buckets[i].items():
                    if cond.mode == "similar":
                        if bk == sbk:
                            bucket.add_match(s.uuid)
                        elif cond.allow_fallback:
                            bucket.add_fallback(s.uuid)
                    elif cond.mode == "dissimilar":
                        if bk != sbk:
                            bucket.add_match(s.uuid)
                        elif cond.allow_fallback:
                            bucket.add_fallback(s.uuid)
                    else:
                        msg = f"Unknown condition mode: {cond.mode}"
                        raise ValueError(msg)

        return all_buckets

    def generate_pairs(self):
        """
        Generate all valid (anchor, pair) combinations based on conditions.

        Returns:
            np.ndarray: Array of shape (n_pairs, 2) with string UUIDs for anchors and pairs.

        Notes:
            - Exhaustive mode (max_pairs_per_anchor=None) can generate up to O(n^2) pairs.
            - Limited mode (int) samples up to `max_pairs_per_anchor` pairs per anchor,
              prioritizing matching samples over fallbacks.
            - Uses cached bucket lookups for efficiency.

        """
        # region: Step 1. Generate ConditionBuckets for each combination of conditions
        all_buckets: list[dict[Any, ConditionBucket]] = self._build_condition_buckets()

        # Cartesian product of (key, sample uuids) from each dict
        bucket_products: dict[tuple, ConditionBucket] = {}
        key_value_groups = [list(d.items()) for d in all_buckets]
        for combo in product(*key_value_groups):
            # bucket_keys are a unique combination (tuple) of all condition buckets: eg (20, 2)
            # cond_buckets are the corresponding ConditionBuckets for those keys: eg (ConditionBucket(key=20, ...), ConditionBucket(key=2, ...))
            bucket_keys, cond_buckets = zip(*combo, strict=True)

            # Get intersection of matching samples across all conditions
            all_matches: list[set[str]] = [cb.matching_samples for cb in cond_buckets]
            joint_matches: set[str] = set.intersection(*map(set, all_matches))

            # Get valid fallbacks
            # Each condition has its own set of matching and fallback samples:
            #   - Joint matches are intersection across all conditions.
            #   - Fallback = (condition-specific fallbacks) U (condition-specific matches)
            #                that were not part of the joint matches.
            #   - Only included if the condition allows fallback.
            all_fb: list[set[str]] = []
            for cond, cb in zip(self.conditions, cond_buckets, strict=True):
                if cond.allow_fallback:
                    fb = (cb.fallback_samples | cb.matching_samples) - joint_matches
                    all_fb.append(fb)
                else:
                    all_fb.append(set())
            joint_fb: set[str] = set.intersection(*all_fb) if all_fb else set()

            # # Get valid fallbacks
            # # Each condition has its own set of matching and fallback samples
            # # Matching samples across all conditions are the interesection of each condition-specific matches
            # # Fallback sample within each condition then become the condition-specific fallbacks samples plus
            # #   the condition-specific matches that are not in the joint matches across all conditions
            # all_fb: list[set[str]] = [
            #     (cb.fallback_samples | cb.matching_samples) - joint_matches for cb in cond_buckets
            # ]
            # joint_fb: set[str] = set.intersection(*map(set, all_fb))

            bucket_products[bucket_keys] = ConditionBucket(
                key=bucket_keys,
                matching_samples=joint_matches,
                fallback_samples=joint_fb,
            )
        # endregion

        # region: Step 2. Generate all pairs (stores sample.uuid values)
        all_pairs: list[tuple[str]] = []
        for s in self.source.samples:
            sample_bucket_keys = self._get_sample_bucket_keys(s)
            cond_bucket = bucket_products[sample_bucket_keys]

            matching_samples = np.fromiter(cond_bucket.matching_samples, dtype="U36")  # UUID is 36 chars long
            fallback_samples = np.fromiter(cond_bucket.fallback_samples, dtype="U36")

            # Exhaustive list of possible pairings for this anchor
            if self.max_pairs_per_anchor is None:
                if len(matching_samples) > 0:
                    selected_matches = np.column_stack(
                        [
                            np.full(len(matching_samples), s.uuid),
                            matching_samples,
                        ],
                    )
                    all_pairs.append(selected_matches)
                if len(fallback_samples) > 0:
                    selected_fb = np.column_stack(
                        [
                            np.full(len(fallback_samples), s.uuid),
                            fallback_samples,
                        ],
                    )
                    all_pairs.append(selected_fb)

            # Select only self.max_pairs_per_anchor (starting with matches)
            elif isinstance(self.max_pairs_per_anchor, int):
                n_from_matches = min(int(self.max_pairs_per_anchor), len(matching_samples))
                n_from_fb = min(
                    int(self.max_pairs_per_anchor) - n_from_matches,
                    len(fallback_samples),
                )
                if n_from_matches > 0:
                    selected_matches = np.column_stack(
                        [
                            np.full(n_from_matches, s.uuid),
                            self.rng.choice(
                                matching_samples,
                                size=n_from_matches,
                                replace=False,
                            ),
                        ],
                    )
                    all_pairs.append(selected_matches)
                if n_from_fb > 0:
                    selected_fb = np.column_stack(
                        [
                            np.full(n_from_fb, s.uuid),
                            self.rng.choice(
                                fallback_samples,
                                size=n_from_fb,
                                replace=False,
                            ),
                        ],
                    )
                    all_pairs.append(selected_fb)

            else:
                msg = f"`max_pairs_per_anchor` must be of type int or None. Received: {type(self.max_pairs_per_anchor)}"
                raise TypeError(msg)
        # endregion

        return np.vstack(all_pairs) if all_pairs else np.empty((0, 2), dtype="U36")

    def build_batches(self) -> list[Batch]:
        """
        Build a list of `Batch` objects from generated sample pairs.

        Returns:
            list[Batch]: List of batches containing paired samples.

        Notes:
            - Each batch contains two roles: "anchor" and "pair".
            - Pair weights are computed by summing condition scores for each anchor/pair.
            - If `drop_last=True`, the final incomplete batch is omitted.
            - Supports shuffling both at the sample-pair level and batch level.

        """
        batches: list[Batch] = []
        all_pairs = self.generate_pairs()

        # Shuffle all samples, if specified
        if self.shuffle:
            self.rng.shuffle(all_pairs, axis=0)

        # Create batches
        for batch_idx, i in enumerate(range(0, len(all_pairs), self.batch_size)):
            anchor_uuids = all_pairs[i : i + self.batch_size, 0]
            anchor_samples = self.source.get_samples_with_uuid(anchor_uuids)

            pair_uuids = all_pairs[i : i + self.batch_size, 1]
            pair_samples = self.source.get_samples_with_uuid(pair_uuids)

            if len(anchor_samples) != len(pair_samples):
                msg = (
                    f"Mismatched batch: {len(anchor_samples)} anchors vs {len(pair_samples)} pairs. "
                    f"Anchor UUIDs: {anchor_uuids}, Pair UUIDs: {pair_uuids}"
                )
                raise RuntimeError(msg)

            pairing_weights = []
            for a, p in zip(anchor_samples.samples, pair_samples.samples, strict=True):
                a_vals = self._get_sample_cond_values(a)
                p_vals = self._get_sample_cond_values(p)
                weight = 0
                for c_idx, cond in enumerate(self.conditions):
                    weight += cond.score(a_vals[c_idx], p_vals[c_idx])
                pairing_weights.append(weight / len(self.conditions))
            if self.drop_last and len(anchor_samples) < self.batch_size:
                if i == 0:
                    warnings.warn(
                        f"The current PairedSampler strategy results in only 1 batch with "
                        f"fewer than `{self.batch_size}` samples and `drop_last=True`. "
                        f"Thus, no batches will be created.",
                        stacklevel=2,
                        category=UserWarning,
                    )
                continue

            b = Batch(
                role_samples={
                    "anchor": anchor_samples,
                    "pair": pair_samples,
                },
                role_sample_weights={
                    "anchor": Data([1] * len(anchor_samples)),
                    "pair": Data(pairing_weights),
                },
                label=batch_idx,
            )
            batches.append(b)

        # Shuffle batch order, if specified
        if self.shuffle:
            self.rng.shuffle(batches)

        return batches
