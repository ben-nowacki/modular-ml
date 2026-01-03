from __future__ import annotations

from typing import Any

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
        show_progress: bool = True,
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
            show_progress (bool, optional):
                Whether to show a progress bar during the batch building process.
                Defaults to True.

        Raises:
            ValueError:
                If both `group_by` and `stratify_by` are provided.

        """
        super().__init__(
            sources=source,
            batch_size=batch_size,
            shuffle=shuffle,
            group_by=group_by,
            group_by_role="default",
            stratify_by=stratify_by,
            stratify_by_role="default",
            strict_stratification=strict_stratification,
            drop_last=drop_last,
            seed=seed,
            show_progress=show_progress,
        )

    def bind_sources(self, source: FeatureSet | FeatureSetView):
        """Instantiates batches via `build_sampled_view()`."""
        if isinstance(source, FeatureSet):
            view = source.to_view()
        elif isinstance(source, FeatureSetView):
            view = source
        else:
            raise TypeError("Sampler source must be a FeatureSet or FeatureSetView.")

        self.sources: dict[str, FeatureSetView] = {view.source.label: view}
        self._sampled = self.build_sampled_view()

    def build_samples(self) -> dict[tuple[str, str], Samples]:
        """
        Construct samples using grouping or stratification logic.

        Returns:
            dict[tuple[str, str], Samples]:
                Mapping of stream labels to Samples objects with attributes
                representing a batch of sample indices and weights.
                The dict key must be a 2-tuple of (stream label, source FeatureSet label).

        """
        if self.sources is None:
            raise RuntimeError("`bind_source` must be called before sampling can occur.")
        src_lbl = next(iter(self.sources.keys()))
        src = self.sources[src_lbl]
        if not isinstance(src, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(src)}"
            raise TypeError(msg)

        # dict key is 2-tuple of stream_label, source_label
        # For single-stream samplers like this one, we use the source label
        # as the stream label
        return {
            (src_lbl, src_lbl): Samples(
                role_indices={"default": src.indices},
                role_weights=None,
            ),
        }

    def __repr__(self):
        return f"SimpleSampler(n_batches={self.num_batches}, batch_size={self.batcher.batch_size})"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return configuration required to reconstruct this sampler.

        Description:
            This *does not* restore the source, only the sampler configurtion.

        Returns:
            dict[str, Any]: Sampler configuration.

        """
        cfg = super().get_config()
        cfg.update({"sampler_name": "SimpleSampler"})
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SimpleSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Sampler configuration.

        Returns:
            SimpleSampler: Unfitted sampler instance.

        """
        if ("sampler_name" not in config) or (config["sampler_name"] != "SimpleSampler"):
            raise ValueError("Invalid config for SimpleSampler.")

        return cls(
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            group_by=config["group_by"],
            stratify_by=config["stratify_by"],
            strict_stratification=config["strict_stratification"],
            drop_last=config["drop_last"],
            seed=config["seed"],
            show_progress=config["show_progress"],
        )
