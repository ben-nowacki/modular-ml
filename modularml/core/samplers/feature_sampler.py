from abc import ABC, abstractmethod

import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.feature_subset import FeatureSubset


class FeatureSampler(ABC):
    def __init__(
        self,
        source: FeatureSet | FeatureSubset | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.source = source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.batches: list[Batch] | None = self.build_batches() if self.source is not None else None

        self._batch_by_uuid: dict[str, Batch] | None = None  # Lazy cache of Batch.batch_id : Batch
        self._batch_by_label: dict[int, Batch] | None = None  # Lazy cache of Batch.index : Batch

    @property
    def batch_ids(self) -> list[str]:
        """
        List of batch UUIDs (`Batch.uuid`).

        Returns:
            list[str]: A list of unique UUIDs, one for each batch.

        Raises:
            RuntimeError: If no source has been bound yet.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_uuid is None:
            self._batch_by_uuid = {b.uuid: b for b in self.batches}
        return list(self._batch_by_uuid.keys())

    @property
    def batch_by_uuid(self) -> dict[str, Batch]:
        """
        Mapping of UUID to Batch.

        Returns:
            dict[str, Batch]: Dictionary of batches keyed by their unique UUID.

        Raises:
            RuntimeError: If no source has been bound yet.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_uuid is None:
            self._batch_by_uuid = {b.uuid: b for b in self.batches}
        return self._batch_by_uuid

    @property
    def batch_by_label(self) -> dict[int, Batch]:
        """
        Mapping of Batch.label → Batch.

        Note: `Batch.label` is a user-defined integer label (usually the order created).
        This is different from indexing with `__getitem__`.

        Returns:
            dict[int, Batch]: Dictionary of batches keyed by their `.label`.

        Raises:
            RuntimeError: If no source has been bound yet.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches are available.")

        if self._batch_by_label is None:
            self._batch_by_label = {b.label: b for b in self.batches}
        return self._batch_by_label

    @property
    def available_roles(self) -> list[str]:
        """
        The set of roles present in batches (e.g., ["default"], or ["anchor", "positive"]).

        Returns:
            list[str] or None: Roles from the first batch, or None if no batches exist.

        """
        if not self.batches:
            return None
        return self.batches[0].available_roles

    def is_bound(self) -> bool:
        """
        Check whether the sampler has a bound source.

        Returns:
            bool: True if a FeatureSet or FeatureSubset is bound and batches are built.

        """
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
        self.batches = self.build_batches()

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
        Retrieve a batch by index or UUID.

        Args:
            key (int | str):
                - If int: the positional index of the batch in the internal list.
                - If str: the `Batch.uuid` of the batch.

        Returns:
            Batch: The requested batch.

        Raises:
            TypeError: If key is not int or str.
            ValueError: If a UUID is provided but not found.

        """
        if isinstance(key, int):
            return self.batches[key]
        if isinstance(key, str):
            return self.get_batch_with_uuid(key)
        msg = f"Invalid key type: {type(key)}. Expected int or str."
        raise TypeError(msg)

    def get_batch_with_uuid(self, batch_uuid: str) -> "Batch":
        """
        Retrieve a batch by its UUID.

        Args:
            batch_uuid (str): The unique UUID of the batch.

        Returns:
            Batch: The batch with the given UUID.

        Raises:
            ValueError: If no batch exists with the given UUID.

        """
        if batch_uuid not in self.batches:
            msg = f"`uuid={batch_uuid}` does not exist in `self.batch_by_uuid`"
            raise ValueError(msg)
        return self.batch_by_uuid[batch_uuid]

    def get_batch_with_label(self, batch_label: int) -> "Batch":
        """
        Retrieve a batch by its `.label`.

        Args:
            batch_label (int): Label assigned during batch construction.

        Returns:
            Batch: The batch with the given label.

        Raises:
            ValueError: If no batch exists with the given label.

        """
        if batch_label not in self.batch_by_label:
            msg = f"`batch_label={batch_label}` does not exist in `self.batch_by_label`"
            raise ValueError(msg)
        return self.batch_by_label[batch_label]

    @abstractmethod
    def build_batches(self) -> list[Batch]:
        """Construct and return a list of batches based on the current sampler configuration."""
