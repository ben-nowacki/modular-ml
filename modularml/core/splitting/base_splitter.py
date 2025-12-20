from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from modularml.core.data.schema_constants import MML_STATE_TARGET
from modularml.utils.serialization.serializable_mixin import SerializableMixin, register_serializable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView


class BaseSplitter(SerializableMixin, ABC):
    """
    Abstract base class for algorithms that derive FeatureSetViews from a FeatureSet.

    Description:
        Defines the core API for all splitters used to partition FeatureSets
        or FeatureSetViews into multiple subsets (e.g., train/val/test).
        Splitters may be called directly on FeatureSetViews, or indirectly
        through a FeatureSet convenience method (e.g., `fs.split_random()`).

        All subclasses must implement `split()`, `get_config()`, and
        `from_config()` methods. The base class handles consistent typing,
        return conventions, and metadata tracking for reproducibility.

    Usage:
        ```python
        splitter = RandomSplitter(ratios={"train": 0.8, "val": 0.2})
        splits = splitter.split(fs.as_view(), return_views=True)
        # or equivalently
        fs.split_random(ratios={"train": 0.8, "val": 0.2})
        ```

    """

    # ================================================
    # Core abstract methods
    # ================================================
    @abstractmethod
    def split(
        self,
        view: FeatureSetView,
        *,
        return_views: bool = True,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Split a FeatureSetView into multiple subsets.

        Description:
            Generates one or more subsets (as either index arrays or FeatureSetViews) \
            derived from the provided FeatureSetView. Subclasses implement the \
            internal logic that determines which sample indices belong to each subset.

            All indices returned by subclasses must be **relative** to the input \
            `FeatureSetView` (i.e., ranging from 0 to len(view)-1).

        Args:
            view (FeatureSetView):
                The input view to partition.
            return_views (bool, optional):
                If True, return a mapping of labels to FeatureSetViews. \
                If False, return a mapping of labels to relative index sequences. \
                Defaults to True.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                A mapping from subset label (e.g., "train", "val", "test") to \
                either a FeatureSetView or an array/list of integer indices.

        """

    # ================================================
    # Convenience methods for subclasses
    # ================================================
    def _return_splits(
        self,
        view: FeatureSetView,
        split_indices: Mapping[str, Sequence[int]],
        *,
        return_views: bool,
    ) -> Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
        """
        Standardized return helper for splitter subclasses.

        Description:
            Converts a dictionary of relative index arrays into either
            FeatureSetViews or raw index arrays depending on `return_views`.

            Subclasses should always produce **relative indices** for the
            provided view. This method ensures they are safely converted
            into new FeatureSetViews via `view.select_rows()`.

        Args:
            view (FeatureSetView):
                The source view from which new views will be derived.
            split_indices (Mapping[str, Sequence[int]]):
                Mapping from label to relative row indices (0 to len(view)-1).
            return_views (bool):
                Whether to return FeatureSetViews or index arrays.

        Returns:
            Mapping[str, FeatureSetView] | Mapping[str, Sequence[int]]:
                Either derived FeatureSetViews or the raw relative index mapping.

        """
        # Validate indices are within the current view
        n = len(view)
        for lbl, idx in split_indices.items():
            if any(i < 0 or i >= n for i in idx):
                msg = (
                    f"Splitter `{type(self).__name__}` produced out-of-range indices "
                    f"for subset '{lbl}'. Expected values in [0, {n - 1}]."
                )
                raise IndexError(msg)

        # Return indices directly
        if not return_views:
            return split_indices

        # FeatureSetView.select_rows constructs new views from itself
        return {label: view.take(rel_indices=idxs, label=label) for label, idxs in split_indices.items()}

    # ============================================
    # Serialization
    # ============================================
    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Serialize this Splitter into a fully reconstructable Python dictionary."""
        ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]):
        """Restore this Splitter configuration in-place from serialized state."""
        ...

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BaseSplitter:
        """Dynamically reconstruct a splitter (including subclasses) from state."""
        from modularml.utils.environment.environment import import_from_path

        splitter_cls = import_from_path(state[MML_STATE_TARGET])

        # Allocate without calling __init__
        obj: BaseSplitter = splitter_cls.__new__(splitter_cls)

        # Restore internal state
        obj.set_state(state)

        return obj


register_serializable(BaseSplitter, kind="sp")
