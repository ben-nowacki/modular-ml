from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data.sample_collection import evaluate_filter_conditions
from modularml.core.data.sample_schema import RAW_VARIANT
from modularml.core.references.featureset_ref import FeatureSetRef
from modularml.core.splitting.splitter_record import SplitterRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.graph.featureset import FeatureSet
    from modularml.core.splitting.base_splitter import BaseSplitter


class SplitMixin:
    """
    Shared mixin providing split-related functionality for both FeatureSet and FeatureSetView.

    Description:
        This mixin unifies split operations across the FeatureSet hierarchy. It allows \
        either a `FeatureSet` or `FeatureSetView` to invoke high-level splitting \
        methods (`split`, `split_random`, `split_by_condition`) without code duplication.

        - When called from a **FeatureSet**, the operation targets the entire \
          SampleCollection.
        - When called from a **FeatureSetView**, the operation applies only to the \
          subset of samples represented by that view.

        Resulting splits are represented as `FeatureSetView` objects and may optionally \
        be registered into the parent FeatureSet's `_splits` and `_split_configs` \
        registries when invoked from a FeatureSet.
    """

    # ==========================================
    # Context resolution
    # ==========================================
    def _get_split_context(self) -> tuple[FeatureSet, bool]:
        """
        Identify the FeatureSet context for a split operation.

        Description:
            Determines the parent FeatureSet and whether the calling instance is a \
            FeatureSetView or a FeatureSet. This allows the same split logic to \
            adapt to both types of objects.

            - If called on a `FeatureSet`, the method returns the instance itself as \
              the source.
            - If called on a `FeatureSetView`, the method returns the parent FeatureSet \
              (`.source`).
            - If the caller is neither, a `TypeError` is raised.

        Returns:
            tuple[FeatureSet, bool]:
                A 2-tuple containing:
                - `FeatureSet`: Reference to the source FeatureSet instance.
                - `bool`: `True` if the caller is a FeatureSetView, `False` otherwise.

        Raises:
            TypeError:
                If the caller is neither a FeatureSet nor a FeatureSetView.

        """
        from modularml.core.data.featureset_view import FeatureSetView
        from modularml.core.graph.featureset import FeatureSet

        # Case 1: called on FeatureSet
        if isinstance(self, FeatureSet):
            source = self
            return source, False

        # Case 2: called on FeatureSetView
        if isinstance(self, FeatureSetView):
            source = self.source
            return source, True

        # Case 3: unknown caller
        msg = f"{self.__class__.__name__} is not a valid split context host. Expected FeatureSet or FeatureSetView."
        raise TypeError(msg)

    # ==========================================
    # Row filtering
    # ==========================================
    def filter(
        self,
        *,
        label: str | None = None,
        variant_to_use: str = RAW_VARIANT,
        **conditions: dict[str, Any | list[Any], Callable],
    ) -> FeatureSetView | None:
        """
        Create a filtered view based on tag, feature, or target conditions.

        Description:
            Filters the FeatureSet data according to key-value conditions \
            applied across all schema domains ("features", "targets", "tags"). \
            Each condition may be:
                - A literal value for equality matching.
                - A list/tuple/set/np.ndarray of allowed values.
                - A callable that takes a NumPy or Arrow array and returns a boolean mask.

            Returns a lightweight :class:`FeatureSetView` referencing the filtered rows \
            of the current FeatureSet (without copying underlying data).

        Args:
            label (str, optional):
                The label to assign to the returned FeatureSetView.
            variant_to_use (str, optional):
                Conditions will only be evaluated on the specified `variant` \
                of each reference column in `conditions`. Defaults to RAW_VARIANT.
            **conditions:
                Mapping of column names (from any domain) to filter criteria.
                Values may be:
                - `scalar`: selects rows where the column equals the value.
                - `sequence`: selects rows where the column is in the given list/set.
                - `callable`: a function that takes a NumPy array and returns a boolean mask.

        Returns:
            FeatureSetView:
                A view of this FeatureSet containing only rows that satisfy \
                all specified conditions. If no rows match, an empty view is returned.

        Raises:
            KeyError:
                If a specified key does not exist in any of the domains.
            TypeError:
                If a condition value type is unsupported.

        Example:
            For a FeatureSet where its samples have the following attributes:
            - FeatureSet.tag_keys() -> 'cell_id', 'group_id', 'pulse_type'
            - FeatureSet.feature_keys() -> 'voltage', 'current',
            - FeatureSet.target_keys() -> 'soh'

            a filter condition can be applied such that:

            - `cell_id` is in [1, 2, 3]
            - `group_id` is greater than 1, and
            - `pulse_type` equals 'charge'.

        ``` python
        FeatureSet.filter(
            cell_id=[1,2,3],
            group_id=(lambda x: x > 1),
            pulse_type='charge',
        )
        ```

        """
        from modularml.core.data.featureset_view import FeatureSetView

        source, is_view = self._get_split_context()
        collection = source.collection
        full_mask = evaluate_filter_conditions(
            collection=collection,
            variant_to_use=variant_to_use,
            **conditions,
        )

        # Restrict mask to current view's indices
        if is_view:
            local_mask = full_mask[self.indices]
            selected_indices = self.indices[np.where(local_mask)[0]]
        else:
            selected_indices = np.where(full_mask)[0]

        if len(selected_indices) == 0:
            warnings.warn(
                f"No samples match filter conditions: {list(conditions.keys())}",
                UserWarning,
                stacklevel=2,
            )

        # Build FeatureSetView using indices
        return FeatureSetView(
            source=source,
            indices=selected_indices,
            label=label or f"{self.label}_filtered",
        )

    # ==========================================
    # Split methods
    # ==========================================
    def split(
        self,
        splitter: BaseSplitter,
        *,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Apply a splitter to this FeatureSe.

        Description:
            Runs the provided `BaseSplitter` instance on a view of the caller. \
            The resulting splits (and splitter config) are optionally registered \
            into the source FeatureSet's `_splits` registry.

        Args:
            splitter (BaseSplitter):
                The splitter instance (e.g., RandomSplitter, ConditionSplitter).
            return_views (bool, optional):
                Whether to return FeatureSetViews or not. Defaults to False.
            register (bool, optional):
                Whether to record the resulting views and splitter config. \
                Defaults to True.

        Returns:
            list[FeatureSetView] | None:
                The created splits are returned only if `return_views=True`.

        """
        # Get context of instance calling this mixin
        source, is_view = self._get_split_context()

        # Choose base view (use self if called on FeatureSetView, otherwise convert FeatureSet to a view)
        base_view = self if is_view else source._as_view()
        if ".fold_" in base_view.label:
            raise NotImplementedError("Splitting of a fold (`base_view.label`) is not supported.")

        # Perform the split (return splits as FeatureSetView instances)
        results: dict[str, FeatureSetView] = splitter.split(base_view, return_views=True)

        # Register splits if requested
        if register:
            # Record FeatureSetView as new split
            for split in list(results.values()):
                source.add_split(split)

            # Record splitter config used
            source._split_configs.append(
                SplitterRecord(
                    splitter_state=splitter.get_state(),
                    applied_to=FeatureSetRef(
                        node=source.label,
                        split=self.label if is_view else None,
                    ),
                ),
            )

        # Return views if requested
        if return_views:
            return results
        return None

    def split_random(
        self,
        ratios: Mapping[str, float],
        *,
        group_by: str | Sequence[str] | None = None,
        seed: int = 13,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Randomly partition this FeatureSet (or FeatureSetView) into subsets.

        Description:
            A convenience wrapper around :class:`RandomSplitter`, which randomly divides \
            the samples of the specified collection into multiple subsets according to \
            user-defined ratios (e.g., `{"train": 0.8, "val": 0.2}`).

            Optionally, one or more tag keys can be provided via `group_by` to ensure \
            that all samples sharing the same tag values (e.g., a common cell ID or batch ID) \
            are placed into the same subset.

        Args:
            ratios (Mapping[str, float]):
                Dictionary mapping subset labels to relative ratios.
                Must sum to 1.0. Example: `{"train": 0.7, "val": 0.2, "test": 0.1}`.
            group_by (str | Sequence[str] | None, optional):
                One or more tag keys to group samples by before splitting.
                If `None`, samples are split individually.
            seed (int, optional):
                Random seed for reproducibility. Defaults to 13.
            return_views (bool, optional):
                Whether to return the resulting FeatureSetViews. Defaults to `False`.
            register (bool, optional):
                Whether to register the resulting splits and splitter configuration in
                `FeatureSet._splits` for future reference. Defaults to `True`.

        Returns:
            list[FeatureSetView] | None:
                The resulting FeatureSetViews (if `return_views=True`).
                Otherwise, returns `None`.

        Example:
            ```python
            fs.split_random(
                ratios={"train": 0.8, "val": 0.2},
                group_by="cell_id",
                seed=42,
            )
            ```

        """
        from modularml.core.splitting.random_splitter import RandomSplitter

        splitter = RandomSplitter(ratios, group_by=group_by, seed=seed)
        return self.split(splitter, return_views=return_views, register=register)

    def split_by_condition(
        self,
        conditions: dict[str, dict[str, Any]],
        *,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Split this FeatureSet (or FeatureSetView) based on logical conditions.

        Description:
            A convenience wrapper around :class:`ConditionSplitter`, which partitions \
            samples into subsets based on user-defined filter expressions.

            Each subset is defined by a dictionary mapping feature, target, or tag keys \
            to condition values, which may be:
            - A literal value for equality matching.
            - A list, tuple, or set of allowed values.
            - A callable predicate ``f(x) -> bool`` that returns a boolean mask.

            For example:
            ```python
            fs.split_by_condition(
                {
                    "low_temp": {"temperature": lambda x: x < 20},
                    "high_temp": {"temperature": lambda x: x >= 20},
                    "cell_5": {"cell_id": 5},
                }
            )
            ```

            **Note:** Overlapping subsets are permitted, but a warning will be issued \
            if any sample satisfies multiple conditions.

        Args:
            conditions (Mapping[str, Mapping[str, Any | Sequence | Callable]]):
                Mapping of subset labels → condition dictionaries. \
                Each condition dictionary maps a key (feature/target/tag name) \
                to a condition (scalar, sequence, or callable).
            return_views (bool, optional):
                Whether to return the resulting FeatureSetViews. Defaults to `False`.
            register (bool, optional):
                Whether to register the resulting splits and splitter configuration in \
                `FeatureSet._splits` for future reference. Defaults to `True`.

        Returns:
            list[FeatureSetView] | None:
                The resulting FeatureSetViews (if `return_views=True`). \
                Otherwise, returns `None`.

        Example:
            ```python
            fs.split_by_condition(
                {
                    "train": {"cell_type": "A"},
                    "test": {"cell_type": "B"},
                }
            )
            ```

        """
        from modularml.core.splitting.condition_splitter import ConditionSplitter

        splitter = ConditionSplitter(**conditions)
        return self.split(splitter, return_views=return_views, register=register)
