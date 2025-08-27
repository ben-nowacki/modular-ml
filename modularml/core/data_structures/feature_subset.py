import weakref
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union

from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.splitters.splitter import BaseSplitter

if TYPE_CHECKING:
    from modularml.core.data_structures.feature_set import FeatureSet
    from modularml.core.data_structures.sample import Sample


class FeatureSubset(SampleCollection):
    """
    A filtered subset of samples from a parent FeatureSet.
    Holds weak reference to parent FeatureSet to avoid circular memory references.
    """
    
    def __init__(self, label: str, parent: "FeatureSet", sample_uuids: List[str], ):
        self.label = label 
        self._parent_ref = weakref.ref(parent)  # weak ref of parent FeatureSet
        self._sample_uuids = set(sample_uuids)
        
        subset_samples = [s for s in parent.samples if s.uuid in self._sample_uuids]
        super().__init__(samples=subset_samples)

        invalid_uuids = [s_uuid for s_uuid in self._sample_uuids if s_uuid not in parent.sample_uuids]
        if invalid_uuids:
            raise ValueError(f"Invalid `Sample.uuid` found in subset: {invalid_uuids}")
    
    @property
    def parent(self) -> 'FeatureSet':
        """Access the parent FeatureSet. Raises error if parent is no longer alive."""
        parent = self._parent_ref()
        if parent is None:
            raise ReferenceError("Parent FeatureSet no longer exists.")
        return parent

    def __repr__(self) -> str:
        return f"FeatureSubset(label='{self.label}', n_samples={len(self)})"


    def is_disjoint_with(self, other: "FeatureSubset") -> bool:
        """
        Checks whether this subset is disjoint from another (i.e., no overlapping samples).

        Args:
            other (FeatureSubset): Another subset to compare against.

        Returns:
            bool: True if the subsets are disjoint (no shared samples), False otherwise.
        """
        return set(self._sample_uuids).isdisjoint(set(other._sample_uuids))
    
    
    
    # ================================================================================
    # Subset Splitting Methods
    # ================================================================================
    def split(self, splitter:BaseSplitter) -> List["FeatureSubset"]:
        """
        Split the current FeatureSubset into multiple FeatureSubsets. 
        *The created splits are automatically added to the parent `FeatureSet.subsets`, \
        in addition to being returned.*

        Args:
            splitter (BaseSplitter): The splitting method.
            
        Returns:
            List[FeatureSubset]: The created subsets.
        """
        
        new_subsets: List[FeatureSubset] = []
        splits = splitter.split(samples=self.samples)
        for k, s_uuids in splits.items():
            subset = FeatureSubset(
                label=k,
                sample_uuids=s_uuids,
                parent=self
            )
            self.parent.add_subset(subset)
            new_subsets.append(subset)
            
        return new_subsets

    
    def split_random(self, ratios: Dict[str, float], seed: int = 42) -> List["FeatureSubset"]:
        """
        Convenience method to split samples randomly based on given ratios.
        This is equivalent to calling `FeatureSet.split(splitter=RandomSplitter(...))`.

        Args:
            ratios (Dict[str, float]): Dictionary mapping subset names to their respective \
                split ratios. E.g., `ratios={'train':0.5, 'test':0.5)`. All values must add \
                to exactly 1.0.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            List[FeatureSubset]: The created subsets.
        """
        from modularml.core.splitters.random_splitter import RandomSplitter
        return self.split(splitter=RandomSplitter(ratios=ratios, seed=seed))

    def split_by_condition(self, **conditions: Dict[str, Dict[str, Any]]) -> List["FeatureSubset"]:
        """
        Convenience method to split samples using condition-based rules.
        This is equivalent to calling `FeatureSet.split(splitter=ConditionSplitter(...))`.
        
        Args:
            **conditions (Dict[str, Dict[str, Any]]): Keyword arguments where each key \
                is a subset name and each value is a dictionary of filter conditions. \
                The filter conditions use the same format as `.filter()` method.

        Examples:
        Below defines three subsets ('low_temp', 'high_temp', and 'cell_5'). The 'low_temp' \
        subset contains all samples with temperatures under 20, the 'high_temp' subsets contains \
        all samples with temperature greater than 20, and the 'cell_5' subset contains all samples \
        where cell_id is 5.
        **Note that subsets can have overlapping samples if the split conditions are not carefully**
        **defined. A UserWarning will be raised when this happens, **
        
        ``` python
            FeatureSet.split_by_condition(
                low_temp={'temperature': lambda x: x < 20},
                high_temp={'temperature': lambda x: x >= 20},
                cell_5={'cell_id': 5}
            )
        ```

        Returns:
            List[FeatureSubset]: The created subsets.
        """
        from modularml.core.splitters.conditon_splitter import ConditionSplitter
        return self.split(splitter=ConditionSplitter(**conditions))
    
        
    # ==========================================
    # State/Config Management Methods
    # ==========================================	
    def get_config(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "sample_uuids": self._sample_uuids,
            "parent_label": self.parent.label if self.parent else None
        }
            
    @classmethod
    def from_config(cls, config: Dict[str, Any], parent:"FeatureSet") -> "FeatureSubset":
        if not parent.label == config['parent_label']:
            raise ValueError(
                f"The provided parent ({parent}) does not match the config definition: "
                f"{parent.label} != {config['parent_label']}."
            )
        
        return cls(
            label=config["label"],
            sample_uuids=config["sample_uuids"],
            parent=parent
        )

