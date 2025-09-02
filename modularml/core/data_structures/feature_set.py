
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import warnings, pickle
from collections import defaultdict
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import copy

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection

from modularml.core.splitters.splitter import BaseSplitter
from modularml.core.data_structures.feature_subset import FeatureSubset
from modularml.core.feature_transforms.feature_transform import FeatureTransform
from modularml.utils.data_format import DataFormat
from modularml.utils.exceptions import SampleLoadError, SubsetOverlapWarning



@dataclass
class TransformRecord:
    fit_spec: str
    apply_spec: str
    transform: FeatureTransform
    mode: Literal['transform', 'inverse_transform']
     
    def get_config(self) -> Dict[str, Any]:
        """
        Return the config (not the full state) of this transform record.
        Does NOT include fitted parameters of the scaler.
        """
        return {
            "fit_spec": self.fit_spec,
            "apply_spec": self.apply_spec,
            "mode": self.mode,
            "transform_config": self.transform.get_config(),
        }
        
    @classmethod
    def from_config(cls, cfg:dict) -> "TransformRecord":
        return cls(
            fit_spec=cfg['fit_spec'],
            apply_spec=cfg['apply_spec'],
            mode=cfg['mode'],
            transform=FeatureTransform.from_config(cfg['transform_config'])
        )
      
      
    def to_serializable(self) -> dict:
        """
        Return the serializable object (ie full state) of this transform record.
        """
        return {
            "fit_spec": self.fit_spec,
            "apply_spec": self.apply_spec,
            "mode": self.mode,
            "transform": self.transform.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, obj: dict) -> "TransformRecord":
        return cls(
            fit_spec=obj['fit_spec'],
            apply_spec=obj['apply_spec'],
            mode=obj['mode'],
            transform=FeatureTransform.from_serializable(obj['transform']),
        )


    def save(self, path: Union[str, Path], overwrite_existing: bool = False):
        """
        Save the full transform record (including fitted state) into a single joblib file.

        Args:
            path (Union[str, Path]): Destination file path (no extension needed).
            overwrite_existing (bool): If True, overwrite existing file. Default = False.
        """
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite_existing:
            raise FileExistsError(f"File already exists at {path}. Use `overwrite_existing=True` to overwrite.")

        joblib.dump(self.to_serializable(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformRecord":
        """
        Load the TransformRecord from a single joblib file
        
        Args:
            path (Union[str, Path]): Path to the saved joblib file.
        
        Returns:
            TransformRecord: Loaded instance.
        """
        path = Path(path).with_suffix('.joblib')
        if not path.exists():
            raise FileNotFoundError(f"No file found at: {path}")

        data = joblib.load(path)
        return cls.from_serializable(data)



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
        self._transform_logs : Dict[str, List[TransformRecord]] = {'features':[], 'targets':[]}
        
    @property
    def available_subsets(self) -> List[str]:
        return list(self.subsets.keys())
    
    @property
    def n_subsets(self) -> int:
        return len(self.available_subsets)
    
    def set_label(self, label:str):
        """Sets FeatureSet.label"""
        if not isinstance(label, str):
            raise TypeError(f"`label` must be a string, not: {label}")
        self.label = label
    
    
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
        

    def filter(self, **conditions: Dict[str, Union[Any, List[Any], Callable]]) -> Optional["FeatureSet"]:
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
            FeatureSet | None: A new FeatureSet containing samples that match all conditions. \
                None is returned if there are no such samples. Note that the samples in the \
                new FeatureSet are a copy of the original, and any modification of one set \
                does not modify the other set.
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

        new_fs = FeatureSet(
            label='filtered',
            samples=[
                copy.deepcopy(self.get_sample_with_uuid(uuid=s_uuid))
                for s_uuid in filtered_sample_uuids
            ]
        )
        return new_fs
        # if returning subset:
        # return FeatureSubset(label='filtered', parent=self, sample_uuids=filtered_sample_uuids)
    
   
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
    
    def split_random(self, ratios: Dict[str, float], group_by: Union[str, List[str]] = None, seed: int = 42) -> List["FeatureSubset"]:
        """
        Convenience method to split samples randomly based on given ratios.
        This is equivalent to calling `FeatureSet.split(splitter=RandomSplitter(...))`.

        Args:
            ratios (Dict[str, float]): Dictionary mapping subset names to their respective \
                split ratios. E.g., `ratios={'train':0.5, 'test':0.5)`. All values must add \
                to exactly 1.0.
            group_by (Union[str, List[str]], optional): Tag key(s) to group samples \
                by before splitting. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            List[FeatureSubset]: The created subsets.
        """
        from modularml.core.splitters.random_splitter import RandomSplitter
        return self.split(splitter=RandomSplitter(ratios=ratios, group_by=group_by, seed=seed))
    
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
    
    
    # ================================================================================
    # Tranform Methods
    # ================================================================================
    def _parse_spec(self, spec: str) -> Tuple[Optional[str], str, Optional[str]]:
            """
            Parse a string like "train.features.voltage" → ("train", "features", "voltage")
            or "features" → (None, "features", None)
            """
            subset, component, key = None, None, None
            
            parts = spec.split(".")
            if len(parts) == 1:
                component = parts[0]
            elif len(parts) == 2:
                subset, component = parts[0], parts[1]
            elif len(parts) == 3:
                subset, component, key = parts[0], parts[1], parts[2]
            else:
                raise ValueError(f"Invalid format '{spec}'. Use 'component', 'subset.component', or 'subset.component.key'.")

            if subset is not None and subset not in self.available_subsets:
                raise ValueError(f"Invalid subset: {subset}. Full spec: {spec}")
            if component not in ['features', 'targets']:
                raise ValueError(f"Invalid component: {component}. Full spec: {spec}")
            if key is not None and key not in list(self.feature_keys) + list(self.target_keys):
                raise ValueError(f"Invalid key: {key}. Full spec: {spec}")
            
            return subset, component, key
        
    def fit_transform(
        self,
        fit: str,
        apply: str,
        transform: FeatureTransform,
    ) -> None:
        """
        Fit the transform to the specified component (fit) and apply to (apply).
        Allowed formats of fit and apply: "component", "subset.component", or "subset.component.key" \
        (applies to all samples).

        Examples:
            >>> FeatureSet.fit_transform(fit="features", apply="features", transform=...)
            >>> FeatureSet.fit_transform(fit="train.features", apply="features", transform=...)
            >>> FeatureSet.fit_transform(fit="train.features.voltage", apply="features.voltage", transform=...)

        Valid components are "features" or "targets".
        """
        if not isinstance(transform, FeatureTransform):
            raise TypeError(f"`transform` must be of type FeatureTransform. Received: {transform}")
    
    
        fit_subset, fit_component, fit_key = self._parse_spec(fit)
        apply_subset, apply_component, apply_key = self._parse_spec(apply)

        if not fit_component == apply_component:
            raise ValueError(f"fit and apply components must match: {fit_component} != {apply_component}")
        if fit_key is not None and not fit_key == apply_key:
            raise ValueError(f"fit and apply keys must match: {fit_key} != {apply_key}")

        # Select samples to fit on and apply to
        fit_samples = self.get_subset(fit_subset).samples if fit_subset else self.samples
        apply_samples = self.get_subset(apply_subset).samples if apply_subset else self.samples

        # Gather all data from the `fit_component` of those samples
        X_fit = None
        if fit_component == 'features':
            if fit_key is not None:
                X_fit = SampleCollection(fit_samples).get_all_features(format=DataFormat.DICT_NUMPY)[fit_key]
            else:
                X_fit = SampleCollection(fit_samples).get_all_features(format=DataFormat.NUMPY)
        elif fit_component == 'targets':
            if fit_key is not None:
                X_fit = SampleCollection(fit_samples).get_all_targets(format=DataFormat.DICT_NUMPY)[fit_key]
            else:
                X_fit = SampleCollection(fit_samples).get_all_targets(format=DataFormat.NUMPY)
        else:
            raise ValueError(f"Invalid fit_component: {fit_component}")

        # Fit FeatureTransform (ensure shape = (n_samples, dim))
        X_fit = np.asarray(X_fit).reshape(len(fit_samples), -1)
        transform.fit(X_fit)

        # Apply to selected component/key in each apply sample
        X_apply = []
        for sample in apply_samples:
            # Sample.features or Sample.targets
            component_dict : Dict[str, Data] = getattr(sample, apply_component)
            if apply_key:
                if apply_key not in component_dict:
                    raise ValueError(f"apply_key ({apply_key}) is missing from Sample.{apply_component}")    
                X_apply.append( component_dict[apply_key].value )
            else:
                X_apply.append( np.vstack([d.value for d in component_dict.values()]) )

        # Ensure shape = (n_samples, ...)
        X_apply = np.asarray(X_apply).reshape(len(apply_samples), -1)

        # Apply transform
        X_transformed = transform.transform(X_apply)

        # Unpack transform back into Samples
        for s, s_trans in zip(apply_samples, X_transformed):
            # Preserve shape: single scaler vs array
            if s_trans.ndim == 0 or s_trans.shape == ():
                new_val = s_trans.value
            else:
                new_val = s_trans
                
            # Mutate specified apply_key
            if apply_key is not None:
                getattr(s, apply_component)[apply_key].value = new_val
            
            # Otherwise, try to split out key dimension
            else:
                all_keys = getattr(s, apply_component).keys()
                new_val = np.atleast_2d(new_val)
                new_val.reshape(len(all_keys), -1)
                for i, k in enumerate(all_keys):
                    getattr(s, apply_component)[k].value = new_val[i]

        
        # Record this transformation
        self._transform_logs[apply_component].append(
            TransformRecord(
                fit_spec=fit, 
                apply_spec=apply, 
                transform=copy.deepcopy(transform),
                mode='transform'
            )
        )
    
    def inverse_transform(self, on: str = 'features'):
        """
        Performs an inverse_transform of the latest applied transform.
        Removes it from the transformation log.

        Args:
            on (str, optional): {'features', 'targets'}. Whether to inverse \
                the last feature transform or last target transform. Defaults \
                to 'features''.
        """
        allowed_ons = ['features', 'targets']
        if not on in allowed_ons:
            raise ValueError(f"`on` must be one of the following: {allowed_ons}. Received: {on}")
        
        if not self._transform_logs[on]:
            raise RuntimeError("No transform to inverse. The transform log is empty.")

        # Pop the last transform
        last_transform = self._transform_logs[on].pop(-1)
        apply_spec = last_transform.apply_spec
        transform = last_transform.transform

        apply_subset, apply_component, apply_key = self._parse_spec(apply_spec)
        apply_samples = self.get_subset(apply_subset).samples if apply_subset else self.samples
        
        # Gather transformed data from samples
        X_apply = []
        for sample in apply_samples:
            component_dict: Dict[str, Data] = getattr(sample, apply_component)
            if apply_key:
                if apply_key not in component_dict:
                    raise ValueError(f"apply_key ({apply_key}) is missing from Sample.{apply_component}")
                X_apply.append(component_dict[apply_key].value)
            else:
                X_apply.append(np.vstack([d.value for d in component_dict.values()]))

        # Ensure shape = (n_samples, ...)
        X_apply = np.asarray(X_apply).reshape(len(apply_samples), -1)

        # Apply inverse transform
        X_restored = transform.inverse_transform(X_apply)
        
        # Write restored values back to Samples
        for s, s_restored in zip(apply_samples, X_restored):
            # Preserve shape: single scaler vs array
            if s_restored.ndim == 0 or s_restored.shape == ():
                new_val = s_restored.value
            else:
                new_val = s_restored

            # Mutate specified apply_key
            if apply_key is not None:
                getattr(s, apply_component)[apply_key].value = new_val
                
            # Otherwise, try to split out key dimension
            else:
                all_keys = getattr(s, apply_component).keys()
                new_val = np.atleast_2d(new_val)
                new_val.reshape(len(all_keys), -1)
                for i, k in enumerate(all_keys):
                    getattr(s, apply_component)[k].value = new_val[i]
        
    def undo_all_transforms(self, on: Optional[str] = None):
        """
        Undo all transforms applied.
        
        Arguments:
            on (str, optional): Can optionally undo all transforms only applied
                to `on`. Must be: 'features', 'targets', or None. If None, all 
                transforms applied to both 'features' and 'targets' will be undone.
                Defaults to None.
        """
        all_ons = []
        if on is None: all_ons = ['features', 'targets']
        elif isinstance(on, str): all_ons = [on, ]
        else:
            raise TypeError(f"`on` must be a str. Received: {on}")
         
        for o in all_ons:
            while len(self._transform_logs[o]) > 0:
                self.inverse_transform(on=o)



    # ==========================================
    # State/Config Management Methods
    # ==========================================	
    def save_samples(self, path: Union[str, Path]):
        """Save the sample data to the specified path."""
        path = Path(path).with_suffix('pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.samples, f)
    
    @staticmethod
    def load_samples(path: Union[str, Path]) -> List[Sample]:
        path = Path(path).with_suffix('pkl')
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_config(self, sample_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Get a serializable configuration of this FeatureSet. 
        
        This does NOT save any data to disk. If `sample_path` is provided, it will 
        be recorded in the config but not written to. Use `save()` for full serialization.

        Args:
            sample_path (Optional[Union[str, Path]]): A reference path to where samples 
                *would* be saved (not actually saved here). If None, assumes in-memory.

        Returns:
            Dict[str, Any]: The config dictionary (e.g., for tracking or JSON export).
        """
        
        return {
            "label": self.label,
            "sample_data": str(sample_path) if sample_path else None,
            "subset_configs": {
                k: v.get_config() for k, v in self.subsets.items()
            },
            "transform_logs": {
                k: [tr.get_config() for tr in v]
                for k, v in self._transform_logs.items()
            },
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureSet":
        sample_path = config.get("sample_data", None)
        if sample_path is None:
            raise ValueError("FeatureSet config is missing 'sample_data' path.")

        try:
            samples = cls.load_samples(sample_path)
        except Exception as e:
            raise SampleLoadError(f"Failed to load saved samples from {sample_path}. {e}")

        fs = cls(label=config["label"], samples=samples)
        for k, subset_cfg in config.get("subset_configs", {}).items():
            fs.add_subset(
                subset=FeatureSubset.from_config(subset_cfg, parent=fs)
            )

        # Restore transformation logs
        for k, log in config.get("transform_logs", {}).items():
            fs._transform_logs[k] = [TransformRecord.from_config(d) for d in log]

        return fs

    def to_serializable(self) -> dict:
        """
        Return the serializable object (ie full state) of this FeatureSet record.
        """
        return {
            "label": self.label,
            "samples": self.samples,
            "subset_configs": {
                k: v.get_config() for k, v in self.subsets.items()
            },
            "transform_logs": {
                k: [tr.to_serializable() for tr in v]
                for k, v in self._transform_logs.items()
            },
        }

    @classmethod
    def from_serializable(cls, obj: dict) -> "FeatureSet":
        # Construct base FeatureSet with required arguments
        fs = cls(label=obj["label"], samples=obj["samples"])

        # Restore subsets
        for k, subset_cfg in obj.get("subset_configs", {}).items():
            fs.add_subset(
                subset=FeatureSubset.from_config(subset_cfg, parent=fs)
            )

        # Restore transform logs
        fs._transform_logs = {
            k: [TransformRecord.from_serializable(tr) for tr in logs]
            for k, logs in obj.get("transform_logs", {}).items()
        }
        return fs

    def save(self, path: Union[str, Path], overwrite_existing:bool=False):
        """
        Save the FeatureSet (samples + config) into a single file using joblib.
        
        Args:
            path (Union[str, Path]): File path to save the FeatureSet.
        """
        path = Path(path).with_suffix('.joblib')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists() and not overwrite_existing:
            raise FileExistsError(f"File already exists at {path}. Use `overwrite_existing=True` to overwrite.")
    
        joblib.dump(self.to_serializable(), path)
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureSet":
        """
        Load the FeatureSet from a single joblib file containing config + samples.
        
        Args:
            path (Union[str, Path]): Path to the saved joblib file.
        
        Returns:
            FeatureSet: Loaded instance.
        """
        path = Path(path).with_suffix('.joblib')
        if not path.exists():
            raise FileNotFoundError(f"No file found at: {path}")

        data = joblib.load(path)
        return cls.from_serializable(data)



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
    
    
            
