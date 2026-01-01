from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.io.protocols import Configurable, Stateful
from modularml.core.sampling.batcher import Batcher
from modularml.utils.data.comparators import deep_equal
from modularml.utils.data.formatting import ensure_list
from modularml.utils.environment.environment import running_in_notebook
from modularml.utils.representation.progress_bars import LazyProgress

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Samples:
    role_indices: dict[str, NDArray[np.int_]]
    role_weights: dict[str, NDArray[np.float32]] | None = None


class BaseSampler(Configurable, Stateful, ABC):
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
        show_progress: bool = True,
    ):
        self.source: FeatureSetView | None = None
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.show_progress = show_progress

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

        # Optional progress bar
        self._progress: LazyProgress | None = None
        self._show_progress = show_progress

        if source is not None:
            self.bind_source(source)

    def __eq__(self, other):
        if not isinstance(other, BaseSampler):
            msg = f"Equality can only be compared between two samplers. Received: {type(other)}."
            raise TypeError(msg)

        # Check config first
        if self.get_config() != other.get_config():
            return False

        # Then check state
        return deep_equal(self.get_state(), other.get_state())

    __hash__ = None

    # =====================================================
    # Properties
    # =====================================================
    @property
    def is_bound(self) -> bool:
        return self.source is not None and self._batches is not None

    @property
    def batches(self) -> list[BatchView]:
        """All batches for this sampler."""
        if self.is_bound and self._batches is None:
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
        self._progress = LazyProgress(
            total=len(self.source) if self.source is not None else None,
            description=f"Sampling ({type(self).__name__})",
            enabled=self._show_progress,
            persist=running_in_notebook(),
        )

        try:
            samples: Samples = self.build_samples()
        finally:
            # Ensure cleanup even if never started
            if self._progress is not None:
                self._progress.close()
                self._progress = None

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
        return {
            "batch_size": self.batcher.batch_size,
            "shuffle": self.batcher.shuffle,
            "group_by": self.batcher.group_by,
            "group_by_role": self.batcher.group_by_role,
            "stratify_by": self.batcher.stratify_by,
            "stratify_by_role": self.batcher.stratify_by_role,
            "strict_stratification": self.batcher.strict_stratification,
            "drop_last": self.batcher.drop_last,
            "seed": self.batcher.seed,
            "show_progress": self.show_progress,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSampler:
        """
        Construct a sampler from configuration.

        Args:
            config (dict[str, Any]): Sampler configuration.

        Returns:
            BaseSampler: Unfitted sampler instance.

        """
        from modularml.samplers import sampler_registry

        if "sampler_name" not in config:
            msg = "Sampler config must store 'sampler_name' if using BaseSampler to instantiate."
            raise KeyError(msg)

        sampler_cls: BaseSampler = sampler_registry[str(config["sampler_name"])]
        return sampler_cls.from_config(config)

    # ================================================
    # Stateful
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """
        Return runtime state (i.e. rng and source)  of the splitter.

        Returns:
            dict[str, Any]: Splitter state.

        """
        # Construct reference of source -> later re-linked via ExperimentContext
        source_cfg: dict[str, Any] | None = None
        if self.source is not None:
            source_cfg = self.source.get_config()

        state = {
            "source_config": source_cfg,
            "batch_configs": None,
        }
        if self._batches is not None:
            bv_configs = []
            for b in self._batches:
                b_cfg = {
                    "role_indices": b.role_indices,
                    "role_indice_weights": b.role_indice_weights,
                }
                bv_configs.append(b_cfg)

            state["batch_configs"] = bv_configs

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore runtime state of the splitter.

        Args:
            state (dict[str, Any]):
                State produced by get_state().

        """
        # If state has source, rebuild view
        if state.get("source_config") is None:
            # Can't set anything without a source
            # Sample is in a config-only runtime state
            return

        # Rebuild source
        view = FeatureSetView.from_config(state["source_config"])

        # If state has batches saved, we can manually set source and batches
        batch_configs = state.get("batch_configs")
        if batch_configs is not None:
            # Construct batch views from psuedo-config
            batches = []
            for b_cfg in batch_configs:
                bv = BatchView(
                    source=view.source,
                    role_indices=b_cfg["role_indices"],
                    role_indice_weights=b_cfg["role_indice_weights"],
                )
                batches.append(bv)
            self._batches = batches

            # Manually set source to avoid rebuilding batches
            self.source = view

        # Otherwise, we should call bind_sources
        else:
            self.bind_source(source=view)

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this Sampler to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the Sampler is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> BaseSampler:
        """
        Load a Sampler from file.

        Args:
            filepath (Path):
                File location of a previously saved Sampler.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            BaseSampler: The reloaded sampler.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
