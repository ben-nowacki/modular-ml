from __future__ import annotations

from typing import TYPE_CHECKING

from modularml.core.sampling.n_sampler import NSampler

if TYPE_CHECKING:
    from modularml.core.data.featureset import FeatureSet
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


class PairedSampler(NSampler):
    """
    Sampler for generating anchor-pair matches using similarity conditions.

    This sampler is a convenience wrapper around `NSampler` for the
    common two-way pairing case. It accepts a single mapping of
    column keys to SimilarityCondition objects and internally delegates
    to `NSampler` using a single role named `"pair"`. The resulting
    batches contain aligned indices under the roles `"anchor"` & `"pair"`
    where each anchor is matched with one or more candidate samples
    that satisfy the provided similarity conditions.
    """

    def __init__(
        self,
        conditions: dict[str, SimilarityCondition],
        source: FeatureSet | FeatureSetView | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        max_pairs_per_anchor: int | None = 3,
        choose_best_only: bool = False,
        group_by: list[str] | None = None,
        group_by_role: str = "anchor",
        stratify_by: list[str] | None = None,
        stratify_by_role: str = "anchor",
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a similarity-based anchor-pair sampler.

        Description:
            ``PairedSampler`` selects, for each anchor sample, a set of
            candidate pair samples that satisfy the provided similarity
            conditions. All keys in ``conditions`` are resolved to concrete
            (domain, key, variant) columns using ``DataReference`` and
            evaluated using their associated SimilarityCondition objects.

            The sampler internally invokes ``NSampler`` with a single role
            named ``"pair"``, meaning each batch will contain:
                - aligned anchor indices under ``"anchor"``
                - aligned partner indices under ``"pair"``
                - per-pair similarity weights

            This sampler is intended for contrastive, metric-learning, or
            supervised-pairing workflows where only two roles are needed.


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

            group_by (list[str], optional):
                FeatureSet key(s) defining grouping behavior.
                Only one grouping strategy can be active at a time.

            group_by_role (str, optional):
                If `group_by=True`, the role on which to draw data for grouping
                must be specified. Defaults to `"anchor"`.

            stratify_by (list[str], optional):
                FeatureSet key(s) defining strata for stratified sampling.
                Conflicts with `group_by`.

            stratify_by_role (str, optional):
                If `stratify_by=True`, the role on which to draw data for stratification
                must be specified. Defaults to `"anchor"`.

            strict_stratification (bool, optional):
                See description above.

            drop_last (bool, optional):
                If True, the final incomplete batch (with fewer than `batch_size`
                pairs) is dropped. Defaults to False.

            seed (int | None, optional):
                Random seed for reproducible shuffling and stochastic selection.

        """
        super().__init__(
            condition_mapping={"pair": conditions},
            source=source,
            batch_size=batch_size,
            shuffle=shuffle,
            max_samples_per_anchor=max_pairs_per_anchor,
            choose_best_only=choose_best_only,
            group_by=group_by,
            group_by_role=group_by_role,
            stratify_by=stratify_by,
            stratify_by_role=stratify_by_role,
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
        )

    def __repr__(self):
        return f"PairedSampler(n_batches={len(self.batches)}, batch_size={self.batcher.batch_size})"
