from abc import ABC, abstractmethod

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.graph.featureset import FeatureSet


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
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.source: FeatureSetView | None = None
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

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
            source = source._as_view()
        if not isinstance(source, FeatureSetView):
            raise TypeError("Sampler source must be FeatureSet or FeatureSetView.")

        self.source = source
        self._batches = self.build_batches()

    # =====================================================
    # Abstract Methods
    # =====================================================
    @abstractmethod
    def build_batches(self) -> list[BatchView]:
        """
        Construct and return a list of BatchView objects.

        A BatchView is a grouping of row indices into roles.
        """
        raise NotImplementedError
