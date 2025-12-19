from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.sampling.batcher import Batcher
from modularml.utils.data_format import ensure_list


@dataclass
class Samples:
    role_indices: dict[str, NDArray[np.int_]]
    role_weights: dict[str, NDArray[np.float32]] | None = None


class BaseSampler(ABC):
    """
    Base class for all samplers.

    Accepts either a FeatureSet or FeatureSetView.
    Always converts to a FeatureSetView internally.
    Produces zero-copy BatchView objects (not materialized batches).
    """

    def __init__(
        self,
        source: FeatureSet | FeatureSetView | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        group_by: list[str] | None = None,
        group_by_role: str = "default",
        stratify_by: list[str] | None = None,
        stratify_by_role: str = "default",
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.source: FeatureSetView | None = None
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        if stratify_by and group_by:
            msg = "Both `group_by` and `stratify_by` cannot be applied at the same."
            raise ValueError(msg)

        self.batcher = Batcher(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            group_by=ensure_list(group_by),
            group_by_role=group_by_role,
            stratify_by=ensure_list(stratify_by),
            stratify_by_role=stratify_by_role,
            strict_stratification=strict_stratification,
            seed=seed,
        )

        self._batches: list[BatchView] | None = None

        if source is not None:
            self.bind_source(source)

    # =====================================================
    # Properties
    # =====================================================
    @property
    def is_bound(self) -> bool:
        return self.source is not None and self._batches is not None

    @property
    def batches(self) -> list[BatchView]:
        """All batches for this sampler."""
        if self._batches is None:
            raise RuntimeError("Sampler has no bound source. Call bind_source().")
        return self._batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        yield from self.batches

    # =====================================================
    # Source Binding (batch instantiation)
    # =====================================================
    def bind_source(self, source: FeatureSet | FeatureSetView):
        """Sets the source and instantiates batches via `build_batches()`."""
        if isinstance(source, FeatureSet):
            source = source.to_view()
        if not isinstance(source, FeatureSetView):
            raise TypeError("Sampler source must be FeatureSet or FeatureSetView.")

        self.source = source
        self._batches = self.build_batches()

    def build_batches(self) -> list[BatchView]:
        """
        Construct and return a list of BatchView objects.

        A BatchView is a grouping of row indices into roles.
        """
        samples: Samples = self.build_samples()
        return self.batcher.batch(
            view=self.source,
            role_indices=samples.role_indices,
            role_weights=samples.role_weights,
        )

    # =====================================================
    # Abstract Methods
    # =====================================================
    @abstractmethod
    def build_samples(self) -> Samples:
        """Construct and return Samples."""
        raise NotImplementedError
