from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.splitters.splitter import BaseSplitter
from modularml.utils.exceptions import SubsetOverlapWarning

if TYPE_CHECKING:
    from collections.abc import Callable

    from modularml.core.data_structures.sample import Sample


class ConditionSplitter(BaseSplitter):
    def __init__(self, **conditions: dict[str, dict[str, Any | list | Callable]]):
        """
        Splits samples into subsets based on user-defined filtering conditions.

        Args:
            conditions (dict[str, dict[str, Union[Any, list[Any], Callable]]]): The outer \
                dict defines keys representing the new subset names. Within each subset, \
                an inner dict defines the filtering conditions to construct the subset. \
                The inner dictionary uses that same format as the `FeatureSet.filter())` \
                method.

        Examples:
        Below defines three subsets ('low_temp', 'high_temp', and 'cell_5'). The 'low_temp' \
        subset contains all samples with temperatures under 20, the 'high_temp' subsets contains \
        all samples with temperature greater than 20, and the 'cell_5' subset contains all samples \
        where cell_id is 5.
        **Note that subsets can have overlapping samples if the split conditions are not carefully**
        **defined. A UserWarning will be raised when this happens, **

        ``` python
            ConditionSplitter(
                low_temp={'temperature': lambda x: x < 20},
                high_temp={'temperature': lambda x: x >= 20},
                cell_5={'cell_id': 5}
            )
        ```

        """
        super().__init__()
        self.conditions = conditions

    def split(self, samples: list[Sample]) -> dict[str, list[str]]:
        """
        Applies the condition-based split on the given samples.

        Args:
            samples (list[Sample]): The list of samples to split.

        Returns:
            dict[str, list[str]]: dictionary mapping subset names to `Sample.uuid`.

        """
        sample_to_subsets = defaultdict(list)  # tracks overlapping samples

        splits = {k: [] for k in self.conditions}
        sample_uuids = np.asarray([s.uuid for s in samples])
        for s_uuid, sample in zip(sample_uuids, samples, strict=True):
            for subset_name, condition_dict in self.conditions.items():
                match = True
                for key, cond in condition_dict.items():
                    value = (
                        sample.get_tags(key)
                        if key in sample.tag_keys
                        else sample.get_features(key)
                        if key in sample.feature_keys
                        else sample.get_targets(key)
                        if key in sample.target_keys
                        else None
                    )
                    if value is None:
                        match = False
                        break

                    if callable(cond):
                        if not cond(value):
                            match = False
                            break
                    elif isinstance(cond, list | tuple | set | np.ndarray):
                        if value not in cond:
                            match = False
                            break
                    elif value != cond:
                        match = False
                        break

                if match:
                    splits[subset_name].append(s_uuid)
                    sample_to_subsets[s_uuid].append(subset_name)

        # Warn if any sample appears in more than one subset
        overlapping = {i: subsets for i, subsets in sample_to_subsets.items() if len(subsets) > 1}
        if overlapping:
            warnings.warn(
                f"\n{len(overlapping)} samples were assigned to multiple subsets. "
                f"Overlap may affect downstream assumptions.\n"
                f"Examples: {dict(list(overlapping.items())[:3])} ...",
                category=SubsetOverlapWarning,
                stacklevel=2,
            )
        return splits

    def get_config(
        self,
    ) -> dict[str, Any]:
        """Returns a configuration to reproduce identical split configurations."""
        cfg = dict(self.conditions)
        cfg["_target_"] = (f"{self.__class__.__module__}.{self.__class__.__name__}",)
        return cfg

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConditionSplitter:
        """Instantiates a ConditionSplitter from config."""
        if config.pop("_target_", None) is not None:  # noqa: SIM102
            if config["_target_"] != f"{cls.__module__}.{cls.__name__}":
                msg = (
                    f"Config _target_ does not match this class: "
                    f"{config['_target_']} != {f'{cls.__module__}.{cls.__name__}'}"
                )
                raise ValueError(msg)

        return cls(**config)
