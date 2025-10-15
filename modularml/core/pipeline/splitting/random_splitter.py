from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.sample_schema import TAGS_COLUMN
from modularml.core.pipeline.splitting.base_splitter import BaseSplitter
from modularml.utils.data_format import ensure_list

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class RandomSplitter(BaseSplitter):
    """
    Randomly splits a FeatureSetView into subsets according to user-specified ratios.

    Description:
        This splitter partitions samples randomly into subsets (e.g., "train", "val", "test") \
        based on the given ratios. Optionally, samples can be grouped by one or more tag keys \
        before splitting, ensuring that all samples from the same group fall into the same subs etc.

        The split operates on **relative indices** within the provided FeatureSetView, \
        preserving full traceability via SAMPLE_ID in the source FeatureSet.

    Example:
        ```python
        splitter = RandomSplitter(ratios={"train": 0.8, "val": 0.2}, group_by="cell_id", seed=42)
        splits = splitter.split(fs_view, return_views=True)
        ```

    """

    def __init__(
        self,
        ratios: Mapping[str, float],
        group_by: str | Sequence[str] | None = None,
        seed: int = 13,
    ):
        """
        Initialize the RandomSplitter.

        Args:
            ratios (Mapping[str, float]):
                Dictionary mapping subset labels to relative ratios. Must sum to 1.0. \
                Example: {"train": 0.7, "val": 0.2, "test": 0.1}.
            group_by (str | Sequence[str] | None, optional):
                One or more tag keys to group samples by before splitting. \
                If None, samples are split individually.
            seed (int, optional):
                Random seed for reproducibility. Default is 13.

        """
        total = float(sum(ratios.values()))
        if not np.isclose(total, 1.0):
            msg = f"ratios must sum to 1.0. Received: {total})."
            raise ValueError(msg)

        self.ratios = dict(ratios)
        self.group_by: list | None = None if group_by is None else ensure_list(group_by)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    # =====================================================
    # Core splitting logic
    # =====================================================
    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Randomly split a FeatureSetView into multiple subsets.

        Description:
            Splits are based on sample order, shuffled using a fixed random seed. \
            If `group_by` is provided, all samples sharing the same tag values \
            are assigned to the same subset.

        Args:
            view (FeatureSetView):
                The input FeatureSetView to partition.
            return_views (bool, optional):
                If True, returns a mapping of labels to FeatureSetViews. \
                If False, returns relative index arrays. Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping of subset label to either FeatureSetViews or index arrays.

        """
        n = len(view)

        # =====================================================
        # Case 1: No grouping
        # =====================================================
        if self.group_by is None:
            rel_indices = np.arange(n)
            self.rng.shuffle(rel_indices)

            boundaries = self._compute_split_boundaries(n)
            split_indices: dict[str, np.ndarray] = {}
            for label, (start, end) in boundaries.items():
                split_indices[label] = rel_indices[start:end]

            return self._return_splits(
                view=view,
                split_indices=split_indices,
                return_views=return_views,
            )

        # =====================================================
        # Case 2: Group by tag(s)
        # =====================================================
        tag_table = view.source.table.column(TAGS_COLUMN)
        tag_table = tag_table.combine_chunks() if tag_table.num_chunks > 1 else tag_table.chunk(0)

        # Extract the tag fields we care about
        group_fields = self.group_by
        group_arrays = [tag_table.field(k) for k in group_fields]

        # Build group keys (tuple per sample)
        group_keys = np.stack([np.array(arr.to_pylist()) for arr in group_arrays], axis=1)
        if group_keys.shape[1] == 1:
            group_keys = group_keys[:, 0]

        # Map group key to indices in this view
        unique_keys, inv = np.unique(group_keys, axis=0, return_inverse=True)
        group_to_indices: dict[int, list[int]] = {}
        for rel_idx, group_id in enumerate(inv):
            group_to_indices.setdefault(group_id, []).append(rel_idx)

        # Shuffle group IDs, not individual samples
        group_ids = np.arange(len(unique_keys))
        self.rng.shuffle(group_ids)

        # Split group IDs according to ratios
        boundaries = self._compute_split_boundaries(len(group_ids))
        split_indices: dict[str, list[int]] = {label: [] for label in self.ratios}

        for label, (start, end) in boundaries.items():
            selected_groups = set(group_ids[start:end])
            for gid in selected_groups:
                split_indices[label].extend(group_to_indices[gid])

        # Convert to sorted arrays (for stable deterministic order)
        split_indices = {label: np.sort(np.array(idxs, dtype=int)) for label, idxs in split_indices.items()}

        return self._return_splits(
            view=view,
            split_indices=split_indices,
            return_views=return_views,
        )

    # =====================================================
    # Utilities
    # =====================================================
    def _compute_split_boundaries(self, n: int) -> dict[str, tuple[int, int]]:
        """Compute index boundaries for each split given n total elements."""
        boundaries = {}
        current = 0
        for i, (label, ratio) in enumerate(self.ratios.items()):
            count = int(ratio * n)
            if i == len(self.ratios) - 1:
                count = n - current
            boundaries[label] = (current, current + count)
            current += count
        return boundaries

    # =====================================================
    # Config serialization
    # =====================================================
    def get_config(self) -> dict[str, Any]:
        """Return configuration to reproduce this splitter."""
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "ratios": self.ratios,
            "group_by": self.group_by,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RandomSplitter:
        """Recreate a RandomSplitter from a configuration dictionary."""
        if "_target_" in config and config["_target_"] != f"{cls.__module__}.{cls.__name__}":
            msg = f"Config _target_ mismatch: expected {cls.__module__}.{cls.__name__}, got {config['_target_']}"
            raise ValueError(msg)

        return cls(
            ratios=config["ratios"],
            group_by=config.get("group_by"),
            seed=config.get("seed", 13),
        )
