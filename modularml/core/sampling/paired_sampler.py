from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.references.data_reference import DataReference
from modularml.core.sampling.base_sampler import BaseSampler

if TYPE_CHECKING:
    from modularml.core.data.sample_collection import SampleCollection
    from modularml.core.graph.featureset import FeatureSet
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


@dataclass(frozen=True)
class SortedColumn:
    values: np.ndarray  # Sorted values
    original_to_sorted_idxs: np.ndarray  # relative indices used to sort the original array


class PairedSampler(BaseSampler):
    def __init__(
        self,
        conditions: dict[str, SimilarityCondition],
        source: FeatureSet | FeatureSetView | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        max_pairs_per_anchor: int | None = 3,
        choose_best_only: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        _summary_.

        Args:
            conditions (dict[str, SimilarityCondition]):
                Mapping from user-defined column specifiers to similarity
                conditions. Each key is resolved using
                :meth:`DataReference.from_string` with the bound FeatureSet label,
                and must ultimately refer to a concrete (domain, key, variant).
                Example keys:
                    - "SOH_PCT"
                    - "features.voltage.raw"
                    - "targets.health_label.transformed"
            source (FeatureSet | FeatureSetView | None, optional):
                Data source to draw samples from. If provided, batches are built
                immediately; otherwise, call :meth:`bind_source` later.
            batch_size (int, optional):
                Number of (anchor, pair) pairs per batch. Defaults to 1.
            shuffle (bool, optional):
                If True, the final list of (anchor, pair) pairs is shuffled before
                batching. Stochastic selection within match categories is also
                controlled by this flag. Defaults to False.
            max_pairs_per_anchor (int | None, optional):
                Maximum number of pairs to generate per anchor:
                - If None: all valid pairs are used.
                - If int: at most `max_pairs_per_anchor` pairs per anchor are
                    created, prioritizing full → partial → non-matches.
                Defaults to 3.
            choose_best_only (bool):
                Instead of generating multiple pairs, select only the \
                highest-score pair per anchor.
            drop_last (bool, optional):
                If True, the final incomplete batch (with fewer than `batch_size`
                pairs) is dropped. Defaults to False.
            seed (int | None, optional):
                Random seed for reproducible shuffling and stochastic selection.

        """
        self.conditions: dict[str, SimilarityCondition] = dict(conditions)
        self.max_pairs_per_anchor = max_pairs_per_anchor
        self.choose_best_only = choose_best_only

        super().__init__(
            source=source,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def build_batches(self) -> list[BatchView]:
        """
        Build (anchor, pair) batches for the bound source.

        Description:
            1. Resolves each condition key into a concrete FeatureSet column \
               using :class:`DataReference` and the active FeatureSet label. \
            2. Extracts the corresponding column arrays from the view's \
               :class:`SampleCollection`.
            3. For each sample as anchor, evaluates all possible candidates and \
               categorizes them (full / partial / non-match) based on per-\
               condition scores and `allow_fallback`.
            4. Selects up to `max_pairs_per_anchor` pairs per anchor, or all \
               valid pairs if `max_pairs_per_anchor is None`.
            5. Aggregates all pairs, optionally shuffles them, and slices into \
               batches of size `batch_size`.
            6. Returns a list of :class:`BatchView` objects with roles \
               `"anchor"` and `"pair"` and pair-specific weights.

        Returns:
            list[BatchView]:
                A list of BatchViews, each describing a batch of (anchor, pair) \
                indices and per-pair weights on the "pair" role.

        Raises:
            RuntimeError:
                If no source has been bound.
            TypeError:
                If the bound source is not a :class:`FeatureSetView`.
            ValueError:
                If column references cannot be resolved for the current source.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches can be built.")
        if not isinstance(self.source, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(self.source)}"
            raise TypeError(msg)

        view: FeatureSetView = self.source
        if len(view) < 2:
            msg = f"PairedSampler requires at least two samples in FeatureSetView. Received: {len(view)}."
            raise ValueError(msg)

        # Generate all (anchor, pair, score) triplets
        anchor_abs_idxs, pair_abs_idxs, pair_scores = self._generate_pairs(view)
        if not anchor_abs_idxs:
            return []

        anchor_abs_idxs = np.asarray(anchor_abs_idxs, dtype=int)
        pair_abs_idxs = np.asarray(pair_abs_idxs, dtype=int)
        pair_scores = np.asarray(pair_scores, dtype=float)

        # Shuffle pairs as a whole, if requested
        if self.shuffle:
            perm = self.rng.permutation(len(anchor_abs_idxs))
            anchor_abs_idxs = anchor_abs_idxs[perm]
            pair_abs_idxs = pair_abs_idxs[perm]
            pair_scores = pair_scores[perm]

        # Build batches
        batches: list[BatchView] = []
        n_pairs = len(anchor_abs_idxs)
        for start in range(0, n_pairs, self.batch_size):
            stop = start + self.batch_size
            a_idx = anchor_abs_idxs[start:stop]
            p_idx = pair_abs_idxs[start:stop]
            w = pair_scores[start:stop]

            if len(a_idx) < self.batch_size and self.drop_last:
                continue

            batches.append(
                BatchView(
                    source=view.source,
                    role_indices={
                        "anchor": a_idx,
                        "pair": p_idx,
                    },
                    role_indice_weights={
                        "anchor": np.ones_like(w, dtype=float),
                        "pair": w,
                    },
                ),
            )

        return batches

    # =====================================================
    # Helpers
    # =====================================================
    def _precompute_numeric_sorted(self, arr: np.ndarray) -> SortedColumn:
        order = np.argsort(arr)
        return SortedColumn(values=arr[order], original_to_sorted_idxs=order)

    def _resolve_columns(self, view: FeatureSetView) -> list[dict[str, Any]]:
        """Returns list[dict[str, Any]]."""
        # Get collection from view
        coll: SampleCollection = view.to_samplecollection()

        specs: list[dict[str, Any]] = []

        # Resolve columns for each condition
        for key_str, cond in self.conditions.items():
            # Infer node/domain/key/variant from string
            ref = DataReference.from_string(
                key_str,
                known_attrs={"node": view.source.label},
                required_attrs=["node", "domain", "key", "variant"],
            )
            if ref.node != view.source.label:
                msg = (
                    "Error resolving condition column. "
                    f"Inferred DataReference '{ref}' does not refer to source '{view.source.label}'."
                )
                raise ValueError(msg)

            # Gather actual data in the column defined by `ref`
            col = coll.get_variant_data(
                domain=ref.domain,
                key=ref.key,
                variant=ref.variant,
                fmt="numpy",
            )
            col = np.asarray(col)

            # Currenlty only support conditions over 1-dimensional data
            col = np.squeeze(col)  # remove singleton dimensions
            if col.ndim != 1:
                msg = (
                    "SimilarityConditions can only be applied to one-dimensional data. "
                    f"Column '{key_str}' has ndim={col.ndim}."
                )
                raise ValueError(msg)

            # We utilize different selection algorithms for numeric and categorical data
            # If numeric, we use a sorted array + binary search
            if np.issubdtype(col.dtype, np.number):
                sorted_col = self._precompute_numeric_sorted(col)
                spec = {
                    "key": key_str,
                    "ref": ref,
                    "cond": cond,
                    "col": col,
                    "sorted": sorted_col,
                    "categorical": False,
                }
            # For categorical, we use binning of unique values
            else:
                # Build reverse lookup: value -> set of relative indices with same value
                cat_map = {}
                for i, v in enumerate(col):
                    cat_map.setdefault(v, []).append(i)
                spec = {
                    "key": key_str,
                    "ref": ref,
                    "cond": cond,
                    "col": col,
                    "cat_map": cat_map,
                    "categorical": True,
                }

            specs.append(spec)

        return specs

    def _find_numeric_matches(
        self,
        anchor_val: Any,
        cond: SimilarityCondition,
        spec: dict[str, Any],
    ) -> tuple[set[int], set[int]]:
        """
        Binary-search for numeric similarity/dissimilarity sets.

        Returns:
            tuple[set[int], set[int]]:
                The relative indices of the column data belonging to the matching and non-matching sets.

        """
        if spec["categorical"]:
            raise ValueError("Cannot compute numeric match on categorical data.")

        # orig_idxs:    [0, 1, 2, 3, 4, 5, ...]
        # after_sort:   [4, 2, 1, 5, 3, 0, ...]  <- this is what is stored in `original_to_sorted_idxs`
        sorted_col: SortedColumn = spec["sorted"]
        vals = sorted_col.values
        idxs = sorted_col.original_to_sorted_idxs
        tol = cond.tolerance

        # Similar values will be within the [a - tol, a + tol] window of the sorted array
        # Similarity: |a - b| <= tol -> window [a - tol, a + tol]
        left = anchor_val - tol
        right = anchor_val + tol

        li = bisect.bisect_left(vals, left)
        ri = bisect.bisect_right(vals, right)

        sim_idxs = set(idxs[li:ri])
        dissim_idxs = set(idxs) - sim_idxs

        overlap = set.intersection(sim_idxs, dissim_idxs)
        if len(overlap):
            msg = f"Samples detected in both matching and non-match groups: {overlap}"
            raise ValueError(msg)

        # Return (matches, non-matches)
        if cond.mode == "similar":
            return sim_idxs, dissim_idxs

        # Debugging that values are unique in match vs non-match
        # print(f"anchor_val: {anchor_val}")
        # print("sim_vals:", {spec["col"][i] for i in sim_idxs})
        # print("dissim_vals:", {spec["col"][i] for i in dissim_idxs})
        return dissim_idxs, sim_idxs

    def _find_categorical_matches(
        self,
        anchor_val: Any,
        cond: SimilarityCondition,
        spec: dict[str, Any],
    ) -> tuple[set[int], set[int]]:
        """
        Categorical matches must be exact.

        Returns:
            tuple[set[int], set[int]]:
                The relative indices of the column data belonging to the matching and non-matching sets.

        """
        if not spec["categorical"]:
            raise ValueError("Cannot compute categorical match on numeric data.")

        vals = spec["col"]
        cat_map = spec["cat_map"]
        sim_idxs = set(cat_map.get(anchor_val, []))
        dissim_idxs = set(range(len(vals))) - sim_idxs

        overlap = set.intersection(sim_idxs, dissim_idxs)
        if len(overlap):
            msg = f"Samples detected in both matching and non-match groups: {overlap}"
            raise ValueError(msg)

        # Return (matches, non-matches)
        if cond.mode == "similar":
            return sim_idxs, dissim_idxs
        return dissim_idxs, sim_idxs

    def _compute_scores(self, anchor_idx: int, cand_list: list[int], specs: dict[str, Any]) -> np.ndarray[float]:
        """All indices must be relative indices."""
        out = []
        for p_idx in cand_list:
            per_cond_score = []
            for spec in specs:
                cond: SimilarityCondition = spec["cond"]
                a_val = spec["col"][anchor_idx]
                b_val = spec["col"][p_idx]
                per_cond_score.append(cond.score(a_val, b_val))
            out.append(float(np.mean(per_cond_score)))
        return np.array(out)

    def _generate_pairs(self, view: FeatureSetView):
        specs = self._resolve_columns(view)
        abs_indices: np.ndarray[int] = view.indices

        # Store selected pairings (anchor + pair store relative indices, score stores scores of pairing)
        anchor_abs_idxs: list[int] = []
        pair_abs_idxs: list[int] = []
        pair_scores: list[float] = []

        for rel_idx in range(len(view)):
            # Collect matches per condition
            per_cond_matches = []
            per_cond_fallbacks = []

            # If a condition does not allow fallback, any sample that do not match
            # that condition cannot be used in other conditions. We use `blocked_rel_idxs`
            # to determine which samples cannot be used
            blocked_rel_idxs: set[int] = set()

            for spec in specs:
                cond: SimilarityCondition = spec["cond"]
                anchor_val = spec["col"][rel_idx]

                # Find matches & non-matches bases on data type
                if not spec["categorical"]:
                    matches, non_matches = self._find_numeric_matches(anchor_val=anchor_val, cond=cond, spec=spec)
                else:
                    matches, non_matches = self._find_categorical_matches(anchor_val=anchor_val, cond=cond, spec=spec)

                # Remove any blocked idxs and anchor
                matches -= blocked_rel_idxs
                matches.discard(rel_idx)
                per_cond_matches.append(matches)

                # Check if fallback is allowed for this cond
                if cond.allow_fallback:
                    per_cond_fallbacks.append(non_matches)
                else:
                    blocked_rel_idxs |= non_matches

            # Compute the set of indices that are globally allowed after
            # enforcing all strict (no-fallback) conditions.
            allowed_idxs: set[int] = set(range(len(view))) - blocked_rel_idxs - {rel_idx}
            if not allowed_idxs:
                # No valid candidates for this anchor
                continue

            # Combine across all conditions
            full_matches: set[int] = set.intersection(*per_cond_matches) & allowed_idxs
            partial_matches: set[int] = (set.union(*per_cond_matches) & allowed_idxs) - full_matches
            non_matches: set[int] = allowed_idxs - full_matches - partial_matches

            # Determine number of pairing to select
            n_select: int = self.max_pairs_per_anchor or (len(view) - 1)

            selected_pairs: list[int] = []
            selected_scores: list[float] = []

            # Search in following order: full_matches -> partial -> non
            for group in (full_matches, partial_matches, non_matches):
                if not group:
                    continue

                # Convert set to list for stable processing
                group_list: list[int] = list(group)

                # Perform full scoring only if requested (expensive)
                if self.choose_best_only:
                    scores: np.ndarray[float] = self._compute_scores(
                        anchor_idx=rel_idx,
                        cand_list=group_list,
                        specs=specs,
                    )
                    order = np.flip(np.argsort(scores))  # higher score = better match
                    group_selected = [group_list[j] for j in order[:n_select]]
                    group_scores = [scores[j] for j in order[:n_select]]
                # Otherwise, use random selection
                else:
                    self.rng.shuffle(group_list)
                    group_selected = group_list[:n_select]
                    group_scores = self._compute_scores(
                        anchor_idx=rel_idx,
                        cand_list=group_selected,
                        specs=specs,
                    )

                # Record selection
                selected_pairs.extend(list(group_selected))
                selected_scores.extend(list(group_scores))

                # Check if need to select from next group
                n_select -= len(group_selected)
                if n_select <= 0:
                    break

            # Records pairing for this single anchor
            for p_rel_idx, score in zip(selected_pairs, selected_scores, strict=True):
                if p_rel_idx == rel_idx:
                    msg = f"Pairing between the same sample is not allowed: {rel_idx} == {p_rel_idx}"
                    raise ValueError(msg)

                # Debugging
                # a_true_val = (
                #     view.to_samplecollection()
                #     .table.slice(rel_idx, 1)
                #     .column("tags")
                #     .combine_chunks()
                #     .field("pulse_soc")
                #     .field("raw")
                #     .to_numpy(zero_copy_only=False),
                # )[0][0]
                # p_true_val = (
                #     view.to_samplecollection()
                #     .table.slice(p_rel_idx, 1)
                #     .column("tags")
                #     .combine_chunks()
                #     .field("pulse_soc")
                #     .field("raw")
                #     .to_numpy(zero_copy_only=False),
                # )[0][0]
                # print(f"anchor: {a_true_val}, \tpair: {p_true_val},\tscore: {score}")

                anchor_abs_idxs.append(abs_indices[rel_idx])
                pair_abs_idxs.append(abs_indices[p_rel_idx])
                pair_scores.append(score)

        return anchor_abs_idxs, pair_abs_idxs, pair_scores
