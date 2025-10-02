from __future__ import annotations

import copy
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import joblib
import numpy as np

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.graph.feature_subset import FeatureSubset
from modularml.core.graph.graph_node import GraphNode
from modularml.core.transforms.feature_transform import FeatureTransform
from modularml.utils.data_format import DataFormat
from modularml.utils.exceptions import SampleLoadError, SubsetOverlapWarning

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from modularml.core.splitters.splitter import BaseSplitter


@dataclass
class TransformRecord:
    fit_spec: str
    apply_spec: str
    transform: FeatureTransform

    def get_config(self) -> dict[str, Any]:
        """
        Return the config (not the full state) of this transform record.

        Does NOT include fitted parameters of the scaler.
        """
        return {
            "fit_spec": self.fit_spec,
            "apply_spec": self.apply_spec,
            "transform_config": self.transform.get_config(),
        }

    @classmethod
    def from_config(cls, cfg: dict) -> TransformRecord:
        return cls(
            fit_spec=cfg["fit_spec"],
            apply_spec=cfg["apply_spec"],
            transform=FeatureTransform.from_config(cfg["transform_config"]),
        )

    def to_serializable(self) -> dict:
        """Return the serializable object (ie full state) of this transform record."""
        return {
            "fit_spec": self.fit_spec,
            "apply_spec": self.apply_spec,
            "transform": self.transform.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, obj: dict) -> TransformRecord:
        return cls(
            fit_spec=obj["fit_spec"],
            apply_spec=obj["apply_spec"],
            transform=FeatureTransform.from_serializable(obj["transform"]),
        )

    def save(self, path: str | Path, *, overwrite_existing: bool = False):
        """
        Save the full transform record (including fitted state) into a single joblib file.

        Args:
            path (Union[str, Path]): Destination file path (no extension needed).
            overwrite_existing (bool): If True, overwrite existing file. Default = False.

        """
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite_existing:
            msg = f"File already exists at {path}. Use `overwrite_existing=True` to overwrite."
            raise FileExistsError(msg)

        joblib.dump(self.to_serializable(), path)

    @classmethod
    def load(cls, path: str | Path) -> TransformRecord:
        """
        Load the TransformRecord from a single joblib file.

        Args:
            path (Union[str, Path]): Path to the saved joblib file.

        Returns:
            TransformRecord: Loaded instance.

        """
        path = Path(path).with_suffix(".joblib")
        if not path.exists():
            msg = f"No file found at: {path}"
            raise FileNotFoundError(msg)

        data = joblib.load(path)
        return cls.from_serializable(data)


@dataclass
class SplitterRecord:
    applied_to: str  # FeatureSet.label or Subset
    split_config: dict[str, Any]  # splitter config


class FeatureSet(SampleCollection, GraphNode):
    """
    Container for structured data.

    Organizes any raw data into a standardized format.
    """

    def __init__(
        self,
        label: str,
        samples: list[Sample],
    ):
        """
        Initiallize a new FeatureSet.

        Args:
            label (str): Name to assign to this FeatureSet
            samples (list[Sample]): list of samples

        """
        SampleCollection.__init__(self, samples=samples)
        GraphNode.__init__(self, label=label, upstream_nodes=None, downstream_nodes=None)

        self._subsets: dict[str, FeatureSubset] = {}

        self._split_configs: list[SplitterRecord] = []
        self._transform_logs: dict[str, list[TransformRecord]] = {"features": [], "targets": []}

    # ==========================================
    # GraphNode Methods
    # ==========================================
    @property
    def allows_upstream_connections(self) -> bool:
        return False  # FeatureSets do not accept inputs

    @property
    def allows_downstream_connections(self) -> bool:
        return True  # FeatureSets can feed into downstream nodes

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return None  # Not applicable

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        # Return shape based on features if available
        return self.feature_shape

    @property
    def max_inputs(self) -> int | None:
        return 0

    # ==========================================
    # FeatureSet Properties & Dunders
    # ==========================================
    @property
    def available_subsets(self) -> list[str]:
        return list(self._subsets.keys())

    @property
    def subsets(self) -> dict[str, FeatureSubset]:
        return self._subsets

    @property
    def n_subsets(self) -> int:
        return len(self.available_subsets)

    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"

    # ==========================================
    # Subset Utilities
    # ==========================================
    def get_subset(self, name: str) -> FeatureSubset:
        """
        Returns the specified subset of this FeatureSet.

        Use `FeatureSet.available_subsets` to view available subset names.

        Args:
            name (str): Subset name to return.

        Returns:
            FeatureSubset: A named view of this FeatureSet.

        """
        if name not in self._subsets:
            msg = f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names."
            raise ValueError(msg)

        return self._subsets[name]

    def add_subset(self, subset: FeatureSubset):
        """
        Adds a new FeatureSubset (view of FeatureSet.samples).

        Args:
            subset (FeatureSubset): The subset to add.

        """
        if subset.label in self._subsets:
            msg = f"Subset label ('{subset.label}') already exists."
            raise ValueError(msg)

        for s in self._subsets.values():
            if not subset.is_disjoint_with(s):
                overlap = set(subset.sample_uuids).intersection(s.sample_uuids)
                warnings.warn(
                    f"\nSubset '{subset.label}' has overlapping samples with existing subset '{s.label}'.\n"
                    f"    (n_overlap = {len(overlap)})\n"
                    f"    Consider checking for disjoint splits or revising your conditions.",
                    SubsetOverlapWarning,
                    stacklevel=2,
                )

        self._subsets[subset.label] = subset

    # def pop_subset(self, name: str) -> "FeatureSubset":
    #     """
    #     Pops the specified subset (removed from FeatureSet and returned).

    #     Args:
    #         name (str): Subset name to pop

    #     Returns:
    #         FeatureSubset: The removed subset.
    #     """

    #     if not name in self._subsets:
    #         raise ValueError(f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names.")

    #     return self._subsets.pop(name)

    # def remove_subset(self, name: str, force:bool = False) -> None:
    #     """
    #     Deletes the specified subset from this FeatureSet.

    #     Args:
    #         name (str): Subset name to remove.
    #         force (bool, optional): Overrides any errors raised during removal.
    #     """

    #     if not name in self._subsets:
    #         raise ValueError(f"`{name}` is not a valid subset. Use `FeatureSet.available_subsets` to view available subset names.")

    #     for sr in self._split_configs:
    #         if sr.applied_to == name:
    #             raise RuntimeError(
    #                 f"You are removing a subset on which a prior split was applied. "
    #                 f"This removes serializability of the applied splitters. "
    #                 f"If this is intentional, set `force=True`."
    #             )
    #         HOW TO CHECK IF SUBSET WAS CREATED WITH OTHER SIBLING SUBSET IN SAME SPLIT?
    #           - this should raise a similar warning as above

    #     del self._subsets[name]

    def clear_subsets(self) -> None:
        """Remove all previously defined subsets."""
        self._subsets.clear()
        self._split_configs = []

    def filter(self, **conditions: dict[str, Any | list[Any], Callable]) -> FeatureSet | None:
        """
        Filter samples using conditions applied to `tags`, `features`, or `targets`.

        Args:
            conditions (dict[str, Union[Any, list[Any], Callable]): Key-value pairs \
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
        ```python
        FeatureSet.filter(cell_id=[1,2,3], group_id=(lambda x: x > 1), pulse_type='charge')
        ```

        Generally, filtering is applied on the attributes of `Sample.tags`, but can \
        also be useful to apply them to the `Sample.target` keys. For example, we \
        might want to filter to a specific state-of-health (soh) range:
        ```python
        # Assuming Sample.targets.keys() returns 'soh', ...
        FeatureSet.filter(soh=lambda x: (x > 85) & (x < 95))
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
                    sample.tags.get(key)
                    if key in sample.tag_keys
                    else sample.features.get(key)
                    if key in sample.feature_keys
                    else sample.targets.get(key)
                    if key in sample.target_keys
                    else None
                )
                if value is None:
                    match = False
                    break

                if callable(condition):
                    if not condition(value):
                        match = False
                        break
                elif isinstance(condition, list | tuple | set | np.ndarray):
                    if value not in condition:
                        match = False
                        break
                elif value != condition:
                    match = False
                    break

            if match:
                filtered_sample_uuids.append(sample.uuid)

        new_fs = FeatureSet(
            label="filtered",
            samples=[copy.deepcopy(self.get_sample_with_uuid(uuid=s_uuid)) for s_uuid in filtered_sample_uuids],
        )
        return new_fs

    # ================================================================================
    # Constructors
    # ================================================================================
    @classmethod
    def from_pandas(
        cls,
        label: str,
        df: pd.DataFrame,
        feature_cols: str | list[str],
        target_cols: str | list[str],
        groupby_cols: str | list[str] | None = None,
        tag_cols: str | list[str] | None = None,
    ) -> FeatureSet:
        """
        Construct a FeatureSet from a pandas DataFrame.

        Args:
            label (str): Label to assign to this FeatureSet.
            df (pd.DataFrame): DataFrame to construct FeatureSet from.
            feature_cols (Union[str, list[str]]): Column name(s) in `df` to use as features.
            target_cols (Union[str, list[str]]): Column name(s) in `df` to use as targets.
            groupby_cols (Union[str, list[str]], optional): If a single feature spans \
                multiple rows in `df`, `groupby_cols` are used to define groups where each group \
                represents a single feature sequence. Defaults to None.
            tag_cols (Union[str, list[str]], optional): Column name(s) corresponding to \
                identifying information that should be retained in the FeatureSet. Defaults to None.

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
            features = {k: Data(df_gb[k].to_numpy() if len(df_gb) > 1 else df_gb[k].iloc[0]) for k in feature_cols}
            targets = {k: Data(df_gb[k].to_numpy() if len(df_gb) > 1 else df_gb[k].iloc[0]) for k in target_cols}
            tags = {k: Data(df_gb[k].to_numpy() if len(df_gb[k].unique()) > 1 else df_gb[k].iloc[0]) for k in tag_cols}
            samples.append(Sample(features=features, targets=targets, tags=tags, label=s_id))

        return cls(label=label, samples=samples)

    from_df = from_pandas  # alias

    @classmethod
    def from_dict(
        cls,
        label: str,
        data: dict[str, list[Any]],
        feature_keys: str | list[str],
        target_keys: str | list[str],
        tag_keys: str | list[str] | None = None,
    ) -> FeatureSet:
        """
        Constructs a FeatureSet from a dictionary.

        Maps dictionary keys to features, targets, and tags.
        All dictionary values should be of the same length (one entry per sample).

        Args:
            label (str): Name to assign to this FeatureSet
            data (dict[str, list[Any]]): Input dictionary. Each key maps to a list of values.
            feature_keys (Union[str, list[str]]): Keys in `data` to be used as features.
            target_keys (Union[str, list[str]]): Keys in `data` to be used as targets.
            tag_keys (Optional[Union[str, list[str]]]): Keys to use as tags. Optional.

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
            msg = f"Inconsistent list lengths in input data: {lengths}"
            raise ValueError(msg)
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
    # Splitting Methods
    # ================================================================================
    def _add_split_config(self, splitter: BaseSplitter, label: str):
        self._split_configs.append(SplitterRecord(applied_to=label, split_config=splitter.get_config()))

    def split(self, splitter: BaseSplitter) -> list[FeatureSubset]:
        """
        Split the current FeatureSet into multiple FeatureSubsets.

        The created splits are automatically added to `FeatureSet.subsets`, in addition to being returned..

        Args:
            splitter (BaseSplitter): The splitting method.

        Returns:
            list[FeatureSubset]: The created subsets.

        """
        new_subsets: list[FeatureSubset] = []
        splits = splitter.split(samples=self.samples)
        for k, s_uuids in splits.items():
            subset = FeatureSubset(label=k, sample_uuids=s_uuids, parent=self)
            self.add_subset(subset)
            new_subsets.append(subset)

        self._add_split_config(splitter=splitter, label=self.label)
        return new_subsets

    def split_random(
        self,
        ratios: dict[str, float],
        group_by: str | list[str] | None = None,
        seed: int = 42,
    ) -> list[FeatureSubset]:
        """
        Convenience method to split samples randomly based on given ratios.

        This is equivalent to calling `FeatureSet.split(splitter=RandomSplitter(...))`.

        Args:
            ratios (dict[str, float]): dictionary mapping subset names to their respective \
                split ratios. E.g., `ratios={'train':0.5, 'test':0.5)`. All values must add \
                to exactly 1.0.
            group_by (Union[str, list[str]], optional): Tag key(s) to group samples \
                by before splitting. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            list[FeatureSubset]: The created subsets.

        """
        from modularml.core.splitters.random_splitter import RandomSplitter  # noqa: PLC0415

        return self.split(splitter=RandomSplitter(ratios=ratios, group_by=group_by, seed=seed))

    def split_by_condition(self, **conditions: dict[str, dict[str, Any]]) -> list[FeatureSubset]:
        """
        Convenience method to split samples using condition-based rules.

        This is equivalent to calling `FeatureSet.split(splitter=ConditionSplitter(...))`.

        Args:
            **conditions (dict[str, dict[str, Any]]): Keyword arguments where each key \
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
            list[FeatureSubset]: The created subsets.

        """
        from modularml.core.splitters.conditon_splitter import ConditionSplitter  # noqa: PLC0415

        return self.split(splitter=ConditionSplitter(**conditions))

    # ================================================================================
    # Tranform Methods
    # ================================================================================
    def _parse_spec(self, spec: str) -> tuple[str | None, str, str | None]:
        """
        Parse a string into a tuple of stirngs using a period as the delimeter.

        Examples:
        - "train.features.voltage" to ("train", "features", "voltage")
        - "features" → (None, "features", None)

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
            msg = f"Invalid format '{spec}'. Use 'component', 'subset.component', or 'subset.component.key'."
            raise ValueError(msg)

        if subset is not None and subset not in self.available_subsets:
            msg = f"Invalid subset: {subset}. Full spec: {spec}"
            raise ValueError(msg)
        if component not in ["features", "targets"]:
            msg = f"Invalid component: {component}. Full spec: {spec}"
            raise ValueError(msg)
        if key is not None and key not in list(self.feature_keys) + list(self.target_keys):
            msg = f"Invalid key: {key}. Full spec: {spec}"
            raise ValueError(msg)

        return subset, component, key

    # def apply_transform_to_collection(
    #     self,
    #     data: SampleCollection,
    #     subset: str,
    #     component: str | None,
    #     *,
    #     inverse: bool = False,
    # ):
    #     """
    #     Apply stored transformations (or their inverse) to a SampleCollection.

    #     Args:
    #         data (SampleCollection): The collection of samples to transform.
    #         subset (str): The subset name (e.g., 'train', 'val') used in the original fit_transform.
    #         component (str | None): The component name ('features' or 'targets'). If None, applies \
    #             (or inverses) transforms on both features and targets.
    #         inverse (bool): Whether to apply the inverse transform.

    #     Returns:
    #         SampleCollection: A new collection with transformed data.

    #     """
    #     if component is not None and component not in {"features", "targets"}:
    #         msg = f"Invalid component: {component}. Must be 'features', 'targets', or None."
    #         raise ValueError(msg)

    #     if component == "features":
    #         for t_record in self._transform_logs["features"]:
    #             # t_record.apply_spec : str
    #             # t_record.transform : FeatureTransform
    #             pass

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
        Valid components are "features" or "targets".

        Examples:
            ```python
            FeatureSet.fit_transform(fit="features", apply="features", transform=...)
            FeatureSet.fit_transform(fit="train.features", apply="features", transform=...)
            FeatureSet.fit_transform(fit="train.features.voltage", apply="features.voltage", transform=...)
            ```

        """
        if not isinstance(transform, FeatureTransform):
            msg = f"`transform` must be of type FeatureTransform. Received: {transform}"
            raise TypeError(msg)

        fit_subset, fit_component, fit_key = self._parse_spec(fit)
        apply_subset, apply_component, apply_key = self._parse_spec(apply)

        if fit_component != apply_component:
            msg = f"fit and apply components must match: {fit_component} != {apply_component}"
            raise ValueError(msg)
        if fit_key is not None and fit_key != apply_key:
            msg = f"fit and apply keys must match: {fit_key} != {apply_key}"
            raise ValueError(msg)

        # Select samples to fit on and apply to
        fit_samples = self.get_subset(fit_subset).samples if fit_subset else self.samples
        apply_samples = self.get_subset(apply_subset).samples if apply_subset else self.samples

        # Gather all data from the `fit_component` of those samples
        x_fit = None
        if fit_component == "features":
            if fit_key is not None:
                x_fit = SampleCollection(fit_samples).get_all_features(fmt=DataFormat.DICT_NUMPY)[fit_key]
            else:
                x_fit = SampleCollection(fit_samples).get_all_features(fmt=DataFormat.NUMPY)
        elif fit_component == "targets":
            if fit_key is not None:
                x_fit = SampleCollection(fit_samples).get_all_targets(fmt=DataFormat.DICT_NUMPY)[fit_key]
            else:
                x_fit = SampleCollection(fit_samples).get_all_targets(fmt=DataFormat.NUMPY)
        else:
            msg = f"Invalid fit_component: {fit_component}"
            raise ValueError(msg)

        # Fit FeatureTransform (ensure shape = (n_samples, dim))
        x_fit = np.asarray(x_fit).reshape(len(fit_samples), -1)
        transform.fit(x_fit)

        # Apply to selected component/key in each apply sample
        x_apply = []
        for sample in apply_samples:
            # Sample.features or Sample.targets
            component_dict: dict[str, Data] = getattr(sample, apply_component)
            if apply_key:
                if apply_key not in component_dict:
                    msg = f"apply_key ({apply_key}) is missing from Sample.{apply_component}"
                    raise ValueError(msg)
                x_apply.append(component_dict[apply_key].value)
            else:
                x_apply.append(np.vstack([d.value for d in component_dict.values()]))

        # Ensure shape = (n_samples, ...)
        x_apply = np.asarray(x_apply).reshape(len(apply_samples), -1)

        # Apply transform
        x_transformed = transform.transform(x_apply)

        # Unpack transform back into Samples
        for s, s_trans in zip(apply_samples, x_transformed, strict=True):
            # Preserve shape: single scaler vs array
            new_val = s_trans.value if s_trans.ndim == 0 or s_trans.shape == () else s_trans

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
            TransformRecord(fit_spec=fit, apply_spec=apply, transform=copy.deepcopy(transform)),
        )

    def inverse_transform(
        self,
        data: SampleCollection,
        component: Literal["features", "targets"],
        *,
        subset: str | None = None,
        which: Literal["all", "last"] = "all",
        inplace: bool = False,
    ) -> SampleCollection | None:
        allowed_components = ["features", "targets"]
        if component not in allowed_components:
            msg = f"`component` must be one of the following: {allowed_components}. Received: {component}"
            raise ValueError(msg)

        if not isinstance(data, SampleCollection):
            msg = f"Data must be of type SampleCollection. Received: {type(data)}"
            raise TypeError(msg)

        # Make copy if don't want to mutate original
        if not inplace:
            data = data.copy()
            self.inverse_transform(
                data=data,
                subset=subset,
                component=component,
                which=which,
                inplace=True,
            )
            return data

        # Get list of applicable transforms
        records: list[TransformRecord] = []
        for record in self._transform_logs[component]:
            r_subset, r_comp, r_key = self._parse_spec(record.apply_spec)
            if r_subset is None or subset is None or r_subset == subset:
                records.append(record)
        # Only take last record if which == 'last'
        if which == "last":
            records = records[-1:]

        # Go through reverse order
        for record in reversed(records):
            r_subset, r_comp, r_key = self._parse_spec(record.apply_spec)

            # Gather transformed data from samples
            x_apply = []
            for sample in data:
                component_dict: dict[str, Data] = getattr(sample, r_comp)
                if r_key:
                    if r_key not in component_dict:
                        msg = f"Key ({r_key}) is missing from Sample.{r_comp}"
                        raise ValueError(msg)
                    x_apply.append(component_dict[r_key].value)
                else:
                    x_apply.append(np.vstack([d.value for d in component_dict.values()]))

            # Ensure shape = (n_samples, ...)
            x_apply = np.asarray(x_apply).reshape(len(data), -1)

            # Apply inverse transform
            x_restored = record.transform.inverse_transform(x_apply)

            # Write restored values back to Samples
            for s, s_restored in zip(data, x_restored, strict=True):
                # Preserve shape: single scaler vs array
                new_val = s_restored.value if s_restored.ndim == 0 or s_restored.shape == () else s_restored

                # Mutate specified key
                if r_key is not None:
                    getattr(s, r_comp)[r_key].value = new_val

                # Otherwise, try to split out key dimension
                else:
                    all_keys = getattr(s, r_comp).keys()
                    new_val = np.atleast_2d(new_val)
                    new_val.reshape(len(all_keys), -1)
                    for i, k in enumerate(all_keys):
                        getattr(s, r_comp)[k].value = new_val[i]

        return None

    def undo_last_transform(self, on: Literal["features", "targets"]):
        allowed_ons = ["features", "targets"]
        if on not in allowed_ons:
            msg = f"`on` must be one of the following: {allowed_ons}. Received: {on}"
            raise ValueError(msg)

        t_record = self._transform_logs[on][-1]
        r_subset, _r_comp, _r_key = self._parse_spec(t_record.apply_spec)

        sample_coll = SampleCollection(self.get_subset(r_subset).samples if r_subset else self.samples)

        # Inverse transform all data in sample_coll (mutated inplace)
        self.inverse_transform(
            data=sample_coll,
            component=on,
            subset=r_subset,
            which="last",
            inplace=True,
        )

        # Remove log
        self._transform_logs[on].pop(-1)

    def undo_all_transforms(self, on: Literal["features", "targets"] | None = None):
        """
        Undo all transforms applied.

        Arguments:
            on (Literal["features", "targets"] | None): Can optionally undo all transforms only applied
                to `on`. Must be: 'features', 'targets', or None. If None, all
                transforms applied to both 'features' and 'targets' will be undone.
                Defaults to None.

        """
        all_ons = [on] if on is not None else ["features", "targets"]
        for o in all_ons:
            while len(self._transform_logs[o]) > 0:
                self.undo_last_transform(on=o)

    # ==========================================
    # State/Config Management Methods
    # ==========================================
    def save_samples(self, path: str | Path):
        """Save the sample data to the specified path."""
        path = Path(path).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(path, "wb") as f:
            pickle.dump(self.samples, f)

    @staticmethod
    def load_samples(path: str | Path) -> list[Sample]:
        path = Path(path).with_suffix(".pkl")
        with Path.open(path, "rb") as f:
            return pickle.load(f)

    def get_config(self, sample_path: str | Path | None = None) -> dict[str, Any]:
        """
        Get a serializable configuration of this FeatureSet.

        This does NOT save any data to disk. If `sample_path` is provided, it will
        be recorded in the config but not written to. Use `save()` for full serialization.

        Args:
            sample_path (Optional[Union[str, Path]]): A reference path to where samples
                *would* be saved (not actually saved here). If None, assumes in-memory.

        Returns:
            dict[str, Any]: The config dictionary (e.g., for tracking or JSON export).

        """
        return {
            "label": self.label,
            "sample_data": str(sample_path) if sample_path else None,
            "subset_configs": {k: v.get_config() for k, v in self._subsets.items()},
            "transform_logs": {k: [tr.get_config() for tr in v] for k, v in self._transform_logs.items()},
            "split_logs": self._split_configs,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSet:
        sample_path = config.get("sample_data")
        if sample_path is None:
            raise ValueError("FeatureSet config is missing 'sample_data' path.")

        try:
            samples = cls.load_samples(sample_path)
        except Exception as e:
            msg = f"Failed to load saved samples from {sample_path}"
            raise SampleLoadError(msg) from e

        fs = cls(label=config["label"], samples=samples)
        for subset_cfg in config.get("subset_configs", {}).values():
            fs.add_subset(subset=FeatureSubset.from_config(subset_cfg, parent=fs))

        # Restore transformation logs
        for k, log in config.get("transform_logs", {}).items():
            fs._transform_logs[k] = [TransformRecord.from_config(d) for d in log]

        # Restore splitters (we don't need to reapply them since subsets are already restored)
        fs._split_configs = config.get("split_logs", [])

        return fs

    def to_serializable(self) -> dict:
        """Return the serializable object (ie full state) of this FeatureSet record."""
        return {
            "label": self.label,
            "samples": self.samples,
            "subset_configs": {k: v.get_config() for k, v in self._subsets.items()},
            "transform_logs": {k: [tr.to_serializable() for tr in v] for k, v in self._transform_logs.items()},
            "split_logs": self._split_configs,
        }

    @classmethod
    def from_serializable(cls, obj: dict) -> FeatureSet:
        # Construct base FeatureSet with required arguments
        fs = cls(label=obj["label"], samples=obj["samples"])

        # Restore subsets
        for subset_cfg in obj.get("subset_configs", {}).values():
            fs.add_subset(subset=FeatureSubset.from_config(subset_cfg, parent=fs))

        # Restore transform logs
        fs._transform_logs = {
            k: [TransformRecord.from_serializable(tr) for tr in logs]
            for k, logs in obj.get("transform_logs", {}).items()
        }

        # Restore splitters (we don't need to reapply them since subsets are already restored)
        fs._split_configs = obj.get("split_logs", [])

        return fs

    def save(self, path: str | Path, *, overwrite_existing: bool = False):
        """
        Save the FeatureSet (samples + config) into a single file using joblib.

        Args:
            path (Union[str, Path]): File path to save the FeatureSet.
            overwrite_existing (bool): Whether to overwrite existing file at path.

        """
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite_existing:
            msg = f"File already exists at {path}. Use `overwrite_existing=True` to overwrite."
            raise FileExistsError(msg)

        joblib.dump(self.to_serializable(), path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        """
        Load the FeatureSet from a single joblib file containing config + samples.

        Args:
            path (Union[str, Path]): Path to the saved joblib file.

        Returns:
            FeatureSet: Loaded instance.

        """
        path = Path(path).with_suffix(".joblib")
        if not path.exists():
            msg = f"No file found at: {path}"
            raise FileNotFoundError(msg)

        data = joblib.load(path)
        return cls.from_serializable(data)
