
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union, TYPE_CHECKING
import warnings, pickle
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection

from modularml.core.splitters.splitter import BaseSplitter
from modularml.core.data_structures.feature_subset import FeatureSubset
from modularml.core.data_structures.feature_transform import FeatureTransform
from modularml.utils.exceptions import SampleLoadError, SubsetOverlapWarning


class FeatureSet(SampleCollection):
    """
    Container for structured data. 
    
    Organizes any raw data into a standardized format.
    """
   
    def __init__(self, label:str, samples: List[Sample]):
        """
        Initiallize a new FeatureSet.

        Args:
            label (str): Name to assign to this FeatureSet
            samples (List[Sample]): List of samples
        """
        super().__init__(samples=samples)
        self.label = str(label)
        self.subsets : Dict[str, FeatureSubset] = {}
        
    @property
    def available_subsets(self):
        return self.subsets.keys()
    
    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"
    


    def get_subset(self, name: str) -> "FeatureSubset":
        """
        Returns the specified subset of this FeatureSet.
        Use `FeatureSet.available_subsets` to view available subset names.
        
        Args:
            name (str): Subset name to return.

        Returns:
            FeatureSubset: A named view of this FeatureSet.
        """
        
        if not name in self.subsets:
            raise ValueError(f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names.")
        
        return self.subsets[name]
    
    def add_subset(self, subset: "FeatureSubset"):
        """Adds a new FeatureSubset (view of FeatureSet.samples).

        Args:
            subset (FeatureSubset): The subset to add.
        """
        if subset.label in self.subsets:
            raise ValueError(f"Subset label ('{subset.label}') already exists.")
        
        for _, s in self.subsets.items():
            if not subset.is_disjoint_with(s):
                overlap = set(subset.sample_ids).intersection(s.sample_ids)
                warnings.warn(
                    f"\nSubset '{subset.label}' has overlapping samples with existing subset '{s.label}'.\n"
                    f"    (n_overlap = {len(overlap)})\n"
                    f"    Consider checking for disjoint splits or revising your conditions.",
                    SubsetOverlapWarning,
                    stacklevel=2
                )
                
        self.subsets[subset.label] = subset
    
    def pop_subset(self, name: str) -> "FeatureSubset":
        """
        Pops the specified subset (removed from FeatureSet and returned).
        
        Args:
            name (str): Subset name to pop 

        Returns:
            FeatureSubset: The removed subset.
        """
        
        if not name in self.subsets:
            raise ValueError(f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names.")
        
        return self.subsets.pop(name)
    
    def remove_subset(self, name: str) -> None:
        """
        Deletes the specified subset from this FeatureSet.
        
        Args:
            name (str): Subset name to remove.s 
        """
        
        if not name in self.subsets:
            raise ValueError(f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names.")
        
        del self.subsets[name]
    
    def clear_subsets(self) -> None:
        """Remove all previously defined subsets."""
        self.subsets.clear()
        

    def filter(self, **conditions: Dict[str, Union[Any, List[Any], Callable]]) -> Optional["FeatureSubset"]:
        """
        Filter samples using conditions applied to `tags`, `features`, or `targets`.

        Args:
            conditions (Dict[str, Union[Any, List[Any], Callable]): Key-value pairs \
                where keys correspond to any attribute of the samples' tags, features, \
                or targets, and values specify the filter condition. Values can be:
                - A literal value (== match)
                - A list/tuple/set/ndarray of values
                - A callable (e.g., lambda x: x < 100)
                    
        Example:
        For a FeatureSet where its samples have the following attributes:
            - Sample.tags.keys() -> 'cell_id', 'group_id', 'pulse_type'
            - Sample.features.keys() -> 'voltage', 'current',
            - Sample.targets.keys() -> 'soh'
        a filter condition can be applied such that:
            - `cell_id` is in [1, 2, 3]
            - `group_id` is greater than 1, and
            - `pulse_type` equals 'charge'.
        ``` python
        >>> FeatureSet.filter(cell_id=[1,2,3], group_id=(lambda x: x > 1), pulse_type='charge')
        ```
        
        Generally, filtering is applied on the attributes of `Sample.tags`, but can \
        also be useful to apply them to the `Sample.target` keys. For example, we \
        might want to filter to a specific state-of-health (soh) range:
        ``` python
        # Assuming Sample.targets.keys() returns 'soh', ...
        >>> FeatureSet.filter(soh=lambda x: (x > 85) & (x < 95))
        ```
        This returns a FeatureSubset that contains a view of the samples in FeatureSet \
        that have a state-of-health between 85% and 95%. 

        Returns:
            FeatureSubset | None: A new subset containing samples that match all conditions. \
                None is returned if there are no such samples.
        """

        filtered_sample_uuids = []
        for sample in self.samples:
            match = True
            for key, condition in conditions.items():
                # Search in tags, then features, then targets
                value = (
                    sample.tags.get(key) if key in sample.tag_keys else
                    sample.features.get(key) if key in sample.feature_keys else
                    sample.targets.get(key) if key in sample.target_keys else
                    None
                )
                if value is None:
                    match = False
                    break

                if callable(condition):
                    if not condition(value):
                        match = False
                        break
                elif isinstance(condition, (list, tuple, set, np.ndarray)):
                    if value not in condition:
                        match = False
                        break
                else:
                    if value != condition:
                        match = False
                        break

            if match:
                filtered_sample_uuids.append(sample.uuid)

        return FeatureSubset(label='filtered', parent=self, sample_uuids=filtered_sample_uuids)

    
   
    # ================================================================================
    # Constructors
    # ================================================================================
    @classmethod
    def from_pandas(
        cls,
        label: str,
        df: pd.DataFrame,
        feature_cols: Union[str, List[str]],
        target_cols: Union[str, List[str]],
        groupby_cols: Optional[Union[str, List[str]]] = None,
        tag_cols: Optional[Union[str, List[str]]] = None,
        feature_transform: Optional["FeatureTransform"] = None,
        target_transform: Optional["FeatureTransform"] = None,
    ) -> 'FeatureSet':
        """
        Construct a FeatureSet from a pandas DataFrame.
        
        Args:
            label (str): Label to assign to this FeatureSet.
            df (pd.DataFrame): DataFrame to construct FeatureSet from.
            feature_cols (Union[str, List[str]]): Column name(s) in `df` to use as features.
            target_cols (Union[str, List[str]]): Column name(s) in `df` to use as targets.
            groupby_cols (Union[str, List[str]], optional): If a single feature spans \
                multiple rows in `df`, `groupby_cols` are used to define groups where each group \
                represents a single feature sequence. Defaults to None.
            tag_cols (Union[str, List[str]], optional): Column name(s) corresponding to \
                identifying information that should be retained in the FeatureSet. Defaults to None.
            feature_transform (FeatureTransform, optional): An optional `FeatureTransform` to apply \
                to the feature(s). Defaults to None.
            target_transform (FeatureTransform, optional): An optional `FeatureTransform` to apply \
                to the target(s). Defaults to None.
        """
        
        # Standardize input args
        feature_cols = [feature_cols] if isinstance(feature_cols, str) else feature_cols
        target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        tag_cols = [tag_cols] if isinstance(tag_cols, str) else tag_cols or []
        groupby_cols = [groupby_cols] if isinstance(groupby_cols, str) else groupby_cols or []

        # Grouping
        groups = None
        if groupby_cols:
            groups = df.groupby(groupby_cols, sort=False)
        else:
            df["_temp_index"] = df.index
            groups = df.groupby("_temp_index", sort=False)

        # Create samples
        samples = []
        for s_id, (_, df_gb) in enumerate(groups):
            features = {
                k: Data(df_gb[k].values if len(df_gb) > 1 else df_gb[k].iloc[0])
                for k in feature_cols
            }
            targets = {
                k: Data(df_gb[k].values if len(df_gb) > 1 else df_gb[k].iloc[0])
                for k in target_cols
            }
            tags = {
                k: Data(df_gb[k].values if len(df_gb[k].unique()) > 1 else df_gb[k].iloc[0])
                for k in tag_cols
            }
            if feature_transform:
                features = feature_transform.apply(features)
            if target_transform:
                targets = target_transform.apply(targets)

            samples.append(Sample(features=features, targets=targets, tags=tags, label=s_id))

        return cls(label=label, samples=samples)
        
    from_df = from_pandas   # alias

    @classmethod
    def from_dict(
        cls,
        label: str,
        data: Dict[str, List[Any]],
        feature_keys: Union[str, List[str]],
        target_keys: Union[str, List[str]],
        tag_keys: Optional[Union[str, List[str]]] = None
    ) -> 'FeatureSet':
        """
        Construct a FeatureSet from a dictionary of column -> list of values.
        Each list should be of the same length (one entry per sample).

        Args:
            label (str): Name to assign to this FeatureSet
            data (Dict[str, List[Any]]): Input dictionary. Each key maps to a list of values.
            feature_keys (Union[str, List[str]]): Keys in `data` to be used as features.
            target_keys (Union[str, List[str]]): Keys in `data` to be used as targets.
            tag_keys (Optional[Union[str, List[str]]]): Keys to use as tags. Optional.

        Returns:
            FeatureSet: A new FeatureSet instance.
        """
        
        # Standardize input args
        feature_keys = [feature_keys] if isinstance(feature_keys, str) else feature_keys
        target_keys = [target_keys] if isinstance(target_keys, str) else target_keys
        tag_keys = [tag_keys] if isinstance(tag_keys, str) else (tag_keys or [])

        # Validate lengths
        lengths = [len(data[k]) for k in feature_keys + target_keys + tag_keys]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent list lengths in input data: {lengths}")
        n_samples = lengths[0]

        # Build list of Sample objects
        samples = []
        for i in range(n_samples):
            features = {k: Data(np.atleast_1d(data[k][i])) for k in feature_keys}
            targets = {k: Data(np.atleast_1d(data[k][i])) for k in target_keys}
            tags = {k: Data(data[k][i]) for k in tag_keys}
            samples.append(Sample(features=features, targets=targets, tags=tags, label=i))

        return cls(label=label, samples=samples)
    


    # ================================================================================
    # Subset Splitting Methods
    # ================================================================================
    def split(self, splitter:BaseSplitter) -> List["FeatureSubset"]:
        """
        Split the current FeatureSet into multiple FeatureSubsets. 
        *The created splits are automatically added to `FeatureSet.subsets`, \
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
            self.add_subset(subset)
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
    def save_samples(self, path: Union[str, Path]):
        """Save the sample data to the specified path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.samples, f)
    
    @staticmethod
    def load_samples(path: Union[str, Path]) -> List[Sample]:
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_config(self, sample_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Get the serialzied configuration of this FeatureSet. Sample data is saved \
            to a file and only the filepath is seriallized.

        Args:
            sample_path (Optional[Union[str, Path]], optional): A path to save the \
                sample data to. If None, the save path is `'./data/{self.label}_samples.pkl'`. \
                Defaults to None.

        Returns:
            Dict[str, Any]: The config dict
        """
        
        # Save samples to a single file
        if sample_path is None:
            sample_path = Path(f"./data/{self.label}_samples.pkl")
        else:
            sample_path = Path(sample_path)
        self.save_samples(sample_path)

        return {
            "label": self.label,
            "sample_data": str(sample_path),
            "subset_configs": {
                k: v.get_config() for k, v in self.subsets.items()
            }
        }
            
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureSubset":
        try:
            samples = cls.load_samples(config['sample_data'])
        except Exception as e: 
            raise SampleLoadError(f"Failed to load saved samples. {e}")
        
        fs = cls(label=config["label"], samples=samples)
        for k, subset_cfg in config.get("subset_configs", {}).items():
            fs.add_subset(
                name=k,
                subset=FeatureSubset.from_config(subset_cfg, parent=fs)
            )
        return fs


    # ================================================================================
    # Visuallization Methods
    # ================================================================================
    def plot_sankey(self):
        """
        Plot a Sankey diagram showing how sample IDs flow across nested FeatureSubsets.
        Subset hierarchy is determined using dot-notation (e.g., 'train.pretrain').
        Samples with multiple subset memberships will show multiple paths.
        """
        import plotly.graph_objects as go
        
        if not self.subsets:
            raise ValueError("No subsets to plot.")

        # Track flow edges: (from_label, to_label) -> set of sample_ids
        flows = defaultdict(set)
        
        # Track all unique nodes for consistent indexing
        all_nodes = set([self.label])  # start from root
        
        # Reverse map: sample_id -> list of subset names it belongs to
        sample_to_subsets = defaultdict(list)
        for subset_name, subset in self.subsets.items():
            for s_uuid in subset.sample_uuids:
                sample_to_subsets[s_uuid].append(subset_name)
            all_nodes.add(subset_name)
        
        # Infer parent from dot-notation (e.g., "train.pretrain" → "train")
        for s_uuid, paths in sample_to_subsets.items():
            for path in paths:
                parts = path.split('.')
                for i in range(len(parts)):
                    parent = self.label if i == 0 else '.'.join(parts[:i])
                    child = '.'.join(parts[:i+1])
                    flows[(parent, child)].add(s_uuid)
                    all_nodes.update([parent, child])

        # Map node name to index
        all_nodes = sorted(all_nodes)
        node_index = {name: i for i, name in enumerate(all_nodes)}
        
        # Build Sankey components
        sources, targets, values, labels = [], [], [], all_nodes
        for (src, tgt), sample_ids in flows.items():
            sources.append(node_index[src])
            targets.append(node_index[tgt])
            values.append(len(sample_ids))  # number of samples flowing
        
        # Plot
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(100,100,200,0.4)"
            )
        )])

        fig.update_layout(title_text="FeatureSet Subset Sankey", font_size=12)
        fig.show()
    
  
    
    # TODO:
    #   - apply transforms in init
    #   - getter/setters for transforms
    #   - method to inverse_transform (maybe as option within the 'get_all_features' and 'get_all_targets')
    #   - improve/rework visuallizations
            
