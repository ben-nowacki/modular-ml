import warnings
from typing import TYPE_CHECKING

from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.samplers.feature_sampler import FeatureSampler

if TYPE_CHECKING:
    from modularml.core.data_structures.sample import Sample

from modularml.core.data_structures.batch import Batch
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.feature_subset import FeatureSubset


class SimpleSampler(FeatureSampler):
    def __init__(
        self,
        source: FeatureSet | FeatureSubset | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        stratify_by: list[str] | None = None,
        group_by: list[str] | None = None,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a sampler that splits a FeatureSet or FeatureSubset into batches.

        Args:
            source (FeatureSet or FeatureSubset, optional):
                Data source to sample from. If provided, batches are built immediately.
                If None, you must call :meth:`bind_source` before using the sampler.
            batch_size (int, optional):
                Number of samples per batch. Defaults to 1.
            shuffle (bool, optional):
                Whether to shuffle samples (and batches) before yielding. Defaults to False.
            stratify_by (list[str] | str, optional):
                One or more `Sample.tags` keys to stratify batches by.
                Ensures each batch has a representative distribution of tag values.
                Example: `stratify_by=["cell_id"]` balances samples across cell IDs.
            group_by (list[str] | str, optional):
                One or more `Sample.tags` keys to group samples into batches.
                Ensures each batch contains only samples from a single group.
                Example: `group_by=["cell_id"]` yields batches grouped by cell.
            drop_last (bool, optional):
                Whether to drop the final batch if it's smaller than `batch_size`.
                Defaults to False.
            seed (int, optional):
                Random seed for reproducible shuffling.

        Raises:
            ValueError: If both `group_by` and `stratify_by` are specified.

        """
        self.stratify_by = (
            stratify_by if isinstance(stratify_by, list) else [stratify_by] if isinstance(stratify_by, str) else None
        )
        self.group_by = group_by if isinstance(group_by, list) else [group_by] if isinstance(group_by, str) else None
        if self.stratify_by is not None and self.group_by is not None:
            msg = "Both `group_by` and `stratify_by` cannot be applied at the same."
            raise ValueError(msg)

        super().__init__(source=source, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, seed=seed)

    def build_batches(self) -> list[Batch]:
        """
        Construct and return a list of batches based on the current sampler configuration.

        Behavior:
            - If `group_by` is set: builds group-specific batches.
            - If `stratify_by` is set: performs stratified interleaving.
            - Otherwise: splits samples sequentially.

        Returns:
            list[Batch]: A list of Batch objects.

        Raises:
            RuntimeError: If no source has been bound.
            ValueError: If stratification is requested but `batch_size < n_strata`.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches can be built.")

        samples: list[Sample] = self.source.samples
        batches: list[Batch] = []
        batch_idx = 0

        # Group samples if group_by is specified
        if self.group_by is not None:
            # Assign sample to their respective groups
            grouped_samples = {}
            for sample in samples:
                key = tuple(sample.tags[k] for k in self.group_by)
                grouped_samples.setdefault(key, []).append(sample)

            for g_key, g_samples in grouped_samples.items():
                # Shuffle samples in this group, if specified
                if self.shuffle:
                    self.rng.shuffle(g_samples)

                # Split group samples into batches
                for i in range(0, len(g_samples), self.batch_size):
                    batch_samples = g_samples[i : i + self.batch_size]
                    if self.drop_last and len(batch_samples) < self.batch_size:
                        if i == 0:
                            warnings.warn(
                                f"The current group (`{g_key}`) has fewer than `{self.batch_size}` samples "
                                f"and `drop_last=True`. No batches will be created with this group.",
                                stacklevel=2,
                                category=UserWarning,
                            )
                        continue

                    batches.append(Batch(role_samples={"default": SampleCollection(batch_samples)}, label=batch_idx))
                    batch_idx += 1

        # stratify_by logic
        elif self.stratify_by is not None:
            # Assign sample to their respective strat groups
            strat_groups = {}
            for sample in samples:
                key = tuple(sample.tags[k] for k in self.stratify_by)
                strat_groups.setdefault(key, []).append(sample)
            if len(strat_groups) > self.batch_size:
                msg = (
                    f"Batch size (n={self.batch_size}) must be greater than or equal "
                    f"to the number of unique strata (n={len(strat_groups)})"
                )
                raise ValueError(msg)

            # Shuffle samples within each stratum
            if self.shuffle:
                for val in strat_groups.values():
                    self.rng.shuffle(val)

            # Flatten in interleaved fashion to ensure class balance
            interleaved_samples = []
            group_iters = {k: iter(v) for k, v in strat_groups.items()}
            group_keys = list(group_iters.keys())
            exhausted = set()
            while len(exhausted) < len(group_keys):
                for k in group_keys:
                    if k in exhausted:
                        continue
                    try:
                        interleaved_samples.append(next(group_iters[k]))
                    except StopIteration:
                        exhausted.add(k)

            # Split interleaved_samples into batches
            for i in range(0, len(interleaved_samples), self.batch_size):
                batch_samples = interleaved_samples[i : i + self.batch_size]
                if self.drop_last and len(batch_samples) < self.batch_size:
                    if i == 0:
                        warnings.warn(
                            f"Stratified sampling results in a batch smaller than batch_size={self.batch_size} "
                            f"and `drop_last=True`. No batches will be created.",
                            stacklevel=2,
                            category=UserWarning,
                        )
                    continue
                batches.append(Batch(role_samples={"default": SampleCollection(batch_samples)}, label=batch_idx))
                batch_idx += 1

        # default logic
        else:
            # Shuffle all samples, if specified
            if self.shuffle:
                self.rng.shuffle(samples)

            # Create batches
            for batch_idx, i in enumerate(range(0, len(samples), self.batch_size)):
                batch_samples = samples[i : i + self.batch_size]

                if self.drop_last and len(batch_samples) < self.batch_size:
                    if i == 0:
                        warnings.warn(
                            f"The current FeatureSampler strategy results in only 1 batch with "
                            f"fewer than `{self.batch_size}` samples and `drop_last=True`. "
                            f"Thus, no batches will be created.",
                            stacklevel=2,
                            category=UserWarning,
                        )
                    continue

                batches.append(Batch(role_samples={"default": SampleCollection(batch_samples)}, label=batch_idx))

        # Shuffle batch order, if specified
        if self.shuffle:
            self.rng.shuffle(batches)

        return batches
