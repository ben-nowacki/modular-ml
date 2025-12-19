from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.schema_constants import DOMAIN_TAGS, REP_RAW
from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.utils.data_format import DataFormat, ensure_list

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class RandomSplitter(BaseSplitter):
    """
    Randomly splits a FeatureSetView into subsets according to user-specified ratios.

    Description:
        This splitter partitions samples randomly into subsets (e.g., "train", "val", "test")
        based on the given ratios. Optionally, samples can be grouped by one or more tag keys
        before splitting, ensuring that all samples from the same group fall into the same subs etc.

        The split operates on **relative indices** within the provided FeatureSetView,
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
                Dictionary mapping subset labels to relative ratios. Must sum to 1.0.
                Example: {"train": 0.7, "val": 0.2, "test": 0.1}.
            group_by (str | Sequence[str] | None, optional):
                One or more tag keys to group samples by before splitting.
                If None, samples are split individually.
            seed (int, optional):
                Random seed for reproducibility. Default is 13.

        """
        total = float(sum(ratios.values()))
        if not np.isclose(total, 1.0):
            msg = f"ratios must sum to 1.0. Received: {total})."
            raise ValueError(msg)

        self.ratios = dict(ratios)
        self.group_by: list[str] | None = None if group_by is None else ensure_list(group_by)
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
        coll = view.source.collection

        # Extract raw tag arrays as numpy, aligned with the FULL FeatureSet
        tag_data: dict[str, np.ndarray] = coll._get_domain_data(
            domain=DOMAIN_TAGS,
            keys=self.group_by,
            fmt=DataFormat.DICT_NUMPY,
            rep=REP_RAW,
            include_rep_suffix=False,
            include_domain_prefix=False,
        )
        # Restrict to the *samples inside this view*
        view_abs_indices = view.indices  # absolute sample indices
        tag_cols_view: list[np.ndarray] = [tag_data[k][view_abs_indices] for k in self.group_by]

        # Build group keys
        # Example:
        #   if grouping by ["cell_id", "cycle_number"]
        #   then group_keys[i] = ("A1", 45)
        group_keys = np.array([tuple(col[i] for col in tag_cols_view) for i in range(n)], dtype=object)

        # Map unique tuple to group ID
        unique_groups, inv = np.unique(group_keys, return_inverse=True)

        # Build mapping: group_id to list of relative indices
        group_to_rel_idxs: dict[int, list[int]] = {}
        for rel_idx, g_id in enumerate(inv):
            group_to_rel_idxs.setdefault(g_id, []).append(rel_idx)

        # Shuffle group IDs, not individual samples
        group_ids = np.arange(len(unique_groups))
        self.rng.shuffle(group_ids)

        # Apply split boundaries at the group level
        boundaries = self._compute_split_boundaries(len(group_ids))
        split_indices: dict[str, list[int]] = {label: [] for label in self.ratios}

        for label, (start, end) in boundaries.items():
            selected = set(group_ids[start:end])
            for g in selected:
                split_indices[label].extend(group_to_rel_idxs[g])

        # Convert & sort
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
        """
        Compute index boundaries for each split given n total elements.

        Args:
            n (int):
                The total number of samples.

        Returns:
            dict[str, tuple[int, int]]:
                A dictionary where keys are subset labels and values are tuples representing
                the start and end indices for each subset.

        """
        boundaries = {}
        current = 0
        for i, (label, ratio) in enumerate(self.ratios.items()):
            count = int(ratio * n)
            if i == len(self.ratios) - 1:
                count = n - current
            boundaries[label] = (current, current + count)
            current += count
        return boundaries

    # ==========================================
    # SerializableMixin
    # ==========================================
    def get_state(self) -> dict[str, Any]:
        """
        Retrieve the internal state of the RandomSplitter.

        Description:
            This method encapsulates the state of the splitter for serialization or deep copying.

        Returns:
            dict[str, Any]:
                A dictionary containing the splitter's state, including version,
                target, ratios, group_by, and seed.

        """
        return {
            "version": "1.0",
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "ratios": self.ratios,
            "group_by": self.group_by,
            "seed": self.seed,
        }

    def set_state(self, state: dict[str, Any]):
        """
        Restore the internal state of the RandomSplitter.

        Description:
            This method allows restoring a splitter from a saved state, ensuring consistency
            and reproducibility.

        Args:
            state (dict[str, Any]):
                The state dictionary to restore from.

        """
        version = state.get("version")
        if version != "1.0":
            msg = f"Unsupported RandomSplitter version: {version}"
            raise NotImplementedError(msg)

        self.ratios = state["ratios"]
        self.group_by = state.get("group_by")
        self.seed = state.get("seed", 13)

    @classmethod
    def from_state(cls, state: dict) -> RandomSplitter:
        """
        Create a RandomSplitter from a saved state.

        Description:
            This classmethod allows creating a new RandomSplitter instance from a saved state,
            useful for loading pre-configured splitters.

        Args:
            state (dict):
                The state dictionary containing the splitter's configuration.

        Returns:
            RandomSplitter:
                A new RandomSplitter instance initialized from the provided state.

        """
        version = state.get("version")
        if version != "1.0":
            msg = f"Unsupported RandomSplitter version: {version}"
            raise NotImplementedError(msg)
        if "_target_" in state and state["_target_"] != f"{cls.__module__}.{cls.__name__}":
            msg = f"State _target_ mismatch: expected {cls.__module__}.{cls.__name__}, got {state['_target_']}"
            raise ValueError(msg)

        return cls(
            ratios=state["ratios"],
            group_by=state.get("group_by"),
            seed=state.get("seed", 13),
        )
