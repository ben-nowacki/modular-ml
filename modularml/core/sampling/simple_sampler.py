from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.sampling.base_sampler import BaseSampler, Samples


class SimpleSampler(BaseSampler):
    def __init__(
        self,
        source: FeatureSet | FeatureSetView | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        group_by: list[str] | None = None,
        stratify_by: list[str] | None = None,
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a sampler that splits a FeatureSet or view into batches.

        Description:
            `SimpleSampler` implements three batching strategies:

            1. **Group-based batching (`group_by`)**
                Samples sharing identical values for the specified columns are \
                placed into the same bucket. Each bucket is then partitioned \
                into batches.

            2. **Stratified batching (`stratify_by`)**
                Samples are grouped into strata defined by column values. \
                Batches are created by interleaving samples from each stratum \
                to maintain balanced representation.

                - If `strict_stratification=True`, batching stops once any \
                  stratum is exhausted (perfectly balanced but may drop samples).
                - If False, batching continues until all samples are consumed \
                  (uses all samples but later batches may be unbalanced).

            3. **Sequential batching**
                Samples are taken in order (optionally shuffled) and cut into \
                fixed-size batches.

            The sampler always returns **zero-copy BatchView objects** that \
            reference the original FeatureSet. BatchViews do *not* materialize \
            data; they only store role → row-index mappings.

        Args:
            source (FeatureSet | FeatureSetView):
                If provided, batches are built immediately; otherwise, call \
                `bind_source()` later.
            batch_size (int):
                Number of samples in each batch.
            shuffle (bool, optional):
                Whether to shuffle samples (and later the resulting batches).
            group_by (list[str], optional):
                FeatureSet key(s) defining grouping behavior. \
                Only one grouping strategy can be active at a time.
            stratify_by (list[str], optional):
                FeatureSet key(s) defining strata for stratified sampling. \
                Conflicts with `group_by`.
            strict_stratification (bool, optional):
                See description above.
            drop_last (bool, optional):
                Drop the final incomplete batch.
            seed (int, optional):
                Random seed for reproducible shuffling.

        Raises:
            ValueError:
                If both `group_by` and `stratify_by` are provided.

        """
        super().__init__(
            source=source,
            batch_size=batch_size,
            shuffle=shuffle,
            group_by=group_by,
            group_by_role="default",
            stratify_by=stratify_by,
            stratify_by_role="default",
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
        )

    def build_samples(self) -> Samples:
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before sampling can occur.")
        if not isinstance(self.source, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(self.source)}"
            raise TypeError(msg)

        return Samples(
            role_indices={"default": self.source.indices},
            role_weights=None,
        )
