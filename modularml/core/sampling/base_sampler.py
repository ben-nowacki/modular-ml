from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import MML_STATE_TARGET
from modularml.core.sampling.batcher import Batcher
from modularml.utils.data.formatting import ensure_list
from modularml.utils.serialization.serializable_mixin import SerializableMixin, register_serializable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from modularml.core.data.batch_view import BatchView


@dataclass
class Samples:
    role_indices: dict[str, NDArray[np.int_]]
    role_weights: dict[str, NDArray[np.float32]] | None = None


class BaseSampler(ABC, SerializableMixin):
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
        if self.is_bound:
            self._batches = self.build_batches()
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

    # ============================================
    # Serialization
    # ============================================
    def get_state(self) -> dict[str, Any]:
        """
        Serialize this Sampler into a fully reconstructable Python dictionary.

        Notes:
            This serializes only the sampler config and source name/id.
            The source data is not saved.

        """
        state: dict[str, Any] = {
            MML_STATE_TARGET: f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "seed": self.seed,
            "batcher": self.batcher.get_config(),
        }
        if self.source is not None:
            state["source"] = {
                "label": self.source.source.label,
                "node_id": self.source.source.node_id,
                "indices": self.source.indices.tolist(),
                "columns": list(self.source.columns),
                "label_view": self.source.label,
            }
        else:
            state["source"] = None

        return state

    def set_state(self, state: dict[str, Any]):
        """
        Restore this Sampler configuration in-place from serialized state.

        This fully restores the Sampler configuration.
        If a source was previously bound, an attempt will be made to re-bind it.
        For this to work, the source must exist in the active ExperimentContext.
        """
        # Restore rng
        self.seed = state.get("seed")
        self.rng = np.random.default_rng(self.seed)

        # Restore batcher
        self.batcher = Batcher.from_config(state["batcher"])

        # Reset runtime fields
        self.source = None
        self._batches = None

        # Attempt to rebind source if present
        src = state.get("source")
        if src is not None:
            from modularml.core.data.featureset_view import FeatureSetView
            from modularml.core.experiment.experiment_context import ExperimentContext

            fs = ExperimentContext.get_node(node_id=src["node_id"])
            if fs is None:
                msg = (
                    f"Cannot restore sampler source '{src['label']}' (node_id={src['node_id']}). FeatureSet not found."
                )
                raise RuntimeError(msg)

            view = FeatureSetView(
                source=fs,
                indices=np.asarray(src["indices"], dtype=int),
                columns=src["columns"],
                label=src["label_view"],
            )

            self.bind_source(view)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BaseSampler:
        """Dynamically reconstruct a sampler (including subclasses) from state."""
        from modularml.utils.environment.environment import import_from_path

        sampler_cls = import_from_path(state[MML_STATE_TARGET])

        # Allocate without calling __init__
        obj: BaseSampler = sampler_cls.__new__(sampler_cls)

        # Restore internal state
        obj.set_state(state)

        return obj


register_serializable(BaseSampler, kind="sm")
