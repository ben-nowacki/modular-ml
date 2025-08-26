from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Any, Callable, Union
import warnings
import numpy as np

from modularml.core.data_structures.sample import Sample
from modularml.core.splitters.splitter import BaseSplitter
from modularml.utils.exceptions import SubsetOverlapWarning


class ConditionSplitter(BaseSplitter):
    def __init__(self, **conditions: Dict[str, Dict[str, Union[Any, List[Any], Callable]]]):
        """
        Splits samples into subsets based on user-defined filtering conditions.

        Args:
            conditions (Dict[str, Dict[str, Union[Any, List[Any], Callable]]]): The outer \
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

    def split(self, samples:List[Sample]) -> Dict[str, List[str]]:
        """
        Applies the condition-based split on the given samples.

        Args:
            samples (List[Sample]): The list of samples to split.

        Returns:
            Dict[str, List[str]]: Dictionary mapping subset names to `Sample.uuid`.
        """
        
        sample_to_subsets = defaultdict(list)   # tracks overlapping samples
        
        splits = {k: [] for k in self.conditions.keys()}
        sample_uuids = np.asarray([
            s.uuid for s in samples
        ])
        for s_uuid, sample in zip(sample_uuids, samples):
            for subset_name, condition_dict in self.conditions.items():
                match = True
                for key, cond in condition_dict.items():
                    value = (
                        sample.get_tags(key) if key in sample.tag_keys else
                        sample.get_features(key) if key in sample.feature_keys else
                        sample.get_targets(key) if key in sample.target_keys else
                        None
                    )
                    if value is None:
                        match = False
                        break
                    
                    if callable(cond):
                        if not cond(value):
                            match = False
                            break
                    elif isinstance(cond, (list, tuple, set, np.ndarray)):
                        if value not in cond:
                            match = False
                            break
                    else:
                        
                        if value != cond:
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
    
    
    