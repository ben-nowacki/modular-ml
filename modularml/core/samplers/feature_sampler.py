import warnings
from typing import TYPE_CHECKING, Union

import numpy as np

from modularml.core.data_structures.sample_collection import SampleCollection

if TYPE_CHECKING:
    from modularml.core.data_structures.sample import Sample

from modularml.core.data_structures.batch import Batch
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.feature_subset import FeatureSubset


class FeatureSampler:
    def __init__(
        self,
        source: Union["FeatureSet", "FeatureSubset"] | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        stratify_by: list[str] | None = None,
        group_by: list[str] | None = None,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initializes a sampler for a FeatureSet or FeatureSubset.

        Args:
            source (FeatureSet or FeatureSubset, optional): The source to sample from. If provided,
                batches are built immediately. If not provided, you must call `bind_source()` before
                using the sampler.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle samples before batching. Defaults to False.
            stratify_by (list[str], optional): One or more `Sample.tags` keys to stratify batches by.
                Ensures each batch has a representative distribution of tag values. For example,
                `stratify_by=["cell_id"]` ensures balanced sampling across different cell IDs.
            group_by (list[str], optional): One or more `Sample.tags` keys to group samples into
                batches. Ensures each batch contains samples only from a single group. For example,
                `group_by=["cell_id"]` yields batches grouped by cell.
            drop_last (bool, optional): Whether to drop the final batch if it's smaller than
                `batch_size`. Defaults to False.
            seed (int, optional): Random seed for reproducible shuffling.

        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.source = source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.stratify_by = (
            stratify_by
            if isinstance(stratify_by, list)
            else [
                stratify_by,
            ]
            if isinstance(stratify_by, str)
            else None
        )
        self.group_by = (
            group_by
            if isinstance(group_by, list)
            else [
                group_by,
            ]
            if isinstance(group_by, str)
            else None
        )
        if self.stratify_by is not None and self.group_by is not None:
            raise ValueError("Both `group_by` and `stratify_by` cannot be applied at the same.")
        self.drop_last = bool(drop_last)

        self.batches: list[Batch] | None = self._build_batches() if self.source is not None else None

        self._batch_by_uuid: dict[str, Batch] | None = None  # Lazy cache of Batch.batch_id : Batch
        self._batch_by_label: dict[int, Batch] | None = None  # Lazy cache of Batch.index : Batch

    @property
    def batch_ids(self) -> list[str]:
        """A list of `Batch.batch_id`."""
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_uuid is None:
            self._batch_by_uuid = {b.uuid: b for b in self.batches}
        return list(self._batch_by_uuid.keys())

    @property
    def batch_by_uuid(self) -> dict[str, Batch]:
        """Contains Batches mapped by the unique `Batch.batch_id` attribute."""
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_uuid is None:
            self._batch_by_uuid = {b.uuid: b for b in self.batches}
        return self._batch_by_uuid

    @property
    def batch_by_label(self) -> dict[int, Batch]:
        """Contains Batches mapped by the user-defined `Batch.index` attribute."""
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_label is None:
            self._batch_by_label = {b.label: b for b in self.batches}
        return self._batch_by_label

    @property
    def available_roles(self) -> list[str]:
        if not self.batches:
            return None
        return self.batches[0].available_roles

    def _build_batches(self) -> list[Batch]:
        """Builds and returns the batches according to shuffle, stratify, or group settings."""
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

    def is_bound(self) -> bool:
        """Returns True if the sampler has a bound source and batches are built."""
        return self.source is not None and self.batches is not None

    def bind_source(self, source: FeatureSet | FeatureSubset):
        """
        Binds a source FeatureSet or FeatureSubset to this sampler and builds batches.

        This method is required if `source` was not provided during initialization.
        It resets and rebuilds internal batches based on the sampler configuration.

        Args:
            source (FeatureSet or FeatureSubset): The data source to bind for sampling.

        """
        self.source = source
        self.batches = self._build_batches()

    def __len__(self):
        """Returns the number of valid batches."""
        return len(self.batches)

    def __iter__(self):
        """
        Generator over valid batches in `self.source`.

        Yields:
            Batch: A single Batch of samples.

        """
        yield from self.batches

    def __getitem__(self, key: int | str) -> Batch:
        """
        Get a Batch by index (position in list) or `Batch.uuid` (str).

        By index is not using the user-defined `Batch.label` attribute, it uses the position of the sample in `Batch._batches`.
        To access Batches by their `.lable` attribute, use the `FeatureSampler.get_batch_with_label()` method.
        """
        if isinstance(key, int):
            return self.batches[key]
        if isinstance(key, str):
            return self.get_batch_with_uuid(key)
        msg = f"Invalid key type: {type(key)}. Expected int or str."
        raise TypeError(msg)

    def get_batch_with_uuid(self, batch_uuid: str) -> "Batch":
        """Return the Batch with `Batch.uuid` == `batch_uuid`, if it exists."""
        if batch_uuid not in self.batches:
            msg = f"`uuid={batch_uuid}` does not exist in `self.batch_by_uuid`"
            raise ValueError(msg)
        return self.batch_by_uuid[batch_uuid]

    def get_batch_with_label(self, batch_label: int) -> "Batch":
        """Return the Batch with `Batch.label` == `batch_label`, if it exists."""
        if batch_label not in self.batch_by_label:
            msg = f"`batch_label={batch_label}` does not exist in `self.batch_by_label`"
            raise ValueError(msg)
        return self.batch_by_label[batch_label]
