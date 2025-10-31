from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

# from modularml.core.graph.feature_subset import FeatureSubset
# from modularml.core.transforms.feature_transform import FeatureTransform
from modularml.components.graph_node import GraphNode, ShapeSpec
from modularml.core.data.sample_collection import SampleCollection, _evaluate_filter_conditions
from modularml.core.data.sample_schema import (
    FEATURES_COLUMN,
    SAMPLE_ID_COLUMN,
    TAGS_COLUMN,
    TARGETS_COLUMN,
    validate_str_list,
)
from modularml.core.pipeline.splitting.base_splitter import BaseSplitter
from modularml.utils.data_conversion import to_numpy
from modularml.utils.data_format import DataFormat, ensure_list
from modularml.utils.exceptions import SampleLoadError, SplitOverlapWarning
from modularml.utils.formatting import flatten_dict_paths
from modularml.utils.pyarrow_data import build_sample_schema_table

if TYPE_CHECKING:
    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_schema import SampleSchema
    # from modularml.core.splitters.splitter import BaseSplitter


# ============================================================================
# Helper Records
# ============================================================================
# @dataclass
# class TransformRecord:
#     fit_spec: str
#     apply_spec: str
#     transform: FeatureTransform

#     def get_config(self) -> dict[str, Any]:
#         return {
#             "fit_spec": self.fit_spec,
#             "apply_spec": self.apply_spec,
#             "transform_config": self.transform.get_config(),
#         }

#     @classmethod
#     def from_config(cls, cfg: dict) -> TransformRecord:
#         return cls(
#             fit_spec=cfg["fit_spec"],
#             apply_spec=cfg["apply_spec"],
#             transform=FeatureTransform.from_config(cfg["transform_config"]),
#         )

#     def to_serializable(self) -> dict:
#         return {
#             "fit_spec": self.fit_spec,
#             "apply_spec": self.apply_spec,
#             "transform": self.transform.to_serializable(),
#         }

#     @classmethod
#     def from_serializable(cls, obj: dict) -> TransformRecord:
#         return cls(
#             fit_spec=obj["fit_spec"],
#             apply_spec=obj["apply_spec"],
#             transform=FeatureTransform.from_serializable(obj["transform"]),
#         )

#     def save(self, path: str | Path, *, overwrite_existing: bool = False):
#         path = Path(path).with_suffix(".joblib")
#         path.parent.mkdir(parents=True, exist_ok=True)
#         if path.exists() and not overwrite_existing:
#             raise FileExistsError(f"File already exists: {path}")
#         joblib.dump(self.to_serializable(), path)

#     @classmethod
#     def load(cls, path: str | Path) -> TransformRecord:
#         path = Path(path).with_suffix(".joblib")
#         if not path.exists():
#             raise FileNotFoundError(f"No file found at: {path}")
#         return cls.from_serializable(joblib.load(path))


# @dataclass
# class SplitterRecord:
#     applied_to: str
#     split_config: dict[str, Any]


# ============================================================================
# FeatureSet
# ============================================================================
class FeatureSet(GraphNode):
    """
    Arrow-backed FeatureSet.

    Acts as the first data node in a ModularML pipeline. Contains data,
    subset definitions, and transformation/splitting logic.
    """

    _default_collection_key: str = "original"
    _transformed_collection_key: str = "transformed"

    def __init__(
        self,
        label: str,
        table: pa.Table,
        schema: SampleSchema | None = None,
    ):
        super().__init__(label=label)
        # Construct new FeatureSet using some original (ie unscaled) data (passed as PyArrow.Table)
        # Table can have its own schema, otherwise uses the default SampleSchema class

        # Stores the underlying data as keyed versions
        self._collections: dict[str, SampleCollection] = {
            self._default_collection_key: SampleCollection(table=table, schema=schema),
        }
        self._active_collection: str | None = self._default_collection_key

        # Stores splits of any sample collection version
        # Outer dict keyed by collection, inner by split name
        self._splits: dict[str, dict[str, FeatureSetView]] = {}

        # self._split_configs: list[SplitterRecord] = []
        # self._transform_logs: dict[str, list[TransformRecord]] = {
        #     "features": [],
        #     "targets": [],
        # }

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
        Construct a FeatureSet from a pandas DataFrame (column-wise storage).

        Description:
            Converts a DataFrame into a column-oriented Arrow table that matches the \
            ModularML `SampleSchema` convention. Each domain (features, targets, tags) \
            becomes a Struct column in the final Arrow table.

            If `groupby_cols` are provided, all rows sharing the same group key are \
            aggregated into a single sample row, with feature and target columns \
            stored as array-valued sequences (e.g., np.ndarray or list).

        Args:
            label (str):
                Name to assign to this FeatureSet.
            df (pd.DataFrame):
                Input pandas DataFrame containing raw experimental or measurement data.
            feature_cols (str | list[str]):
                Column name(s) in `df` to be used as feature variables.
            target_cols (str | list[str]):
                Column name(s) in `df` to be used as target variables.
            groupby_cols (str | list[str] | None, optional):
                One or more column names defining group boundaries. Each group becomes \
                one sample (row) in the final table, with grouped columns stored as lists. \
                If None, each original DataFrame row is treated as a sample.
            tag_cols (str | list[str] | None, optional):
                Column name(s) corresponding to identifying or categorical metadata \
                (e.g., cell ID, protocol, SOC). Defaults to None.

        Returns:
            FeatureSet:
                A new Arrow-backed FeatureSet whose table follows the ModularML \
                SampleSchema convention.

        Example:
            ```python
            fs = FeatureSet.from_pandas(
                label="PulseData",
                df=raw_df,
                feature_cols=["voltage", "current"],
                target_cols=["soh"],
                groupby_cols=["cell_id", "cycle_index"],
                tag_cols=["temperature", "cell_id"],
            )
            ```

        """
        # 1. Standardize input args
        feature_cols = ensure_list(feature_cols)
        target_cols = ensure_list(target_cols)
        tag_cols = ensure_list(tag_cols)
        groupby_cols = ensure_list(groupby_cols)

        # 2. Apply grouping, if defined
        if groupby_cols:
            grouped = df.groupby(groupby_cols, sort=False)
        else:
            # Each row is a separate sample -> pseudo group by index
            df = df.copy()
            df["_temp_index"] = df.index
            grouped = df.groupby("_temp_index", sort=False)

        # 3. Aggregate each group into one “row”
        feature_data: dict[str, list] = {c: [] for c in feature_cols}
        target_data: dict[str, list] = {c: [] for c in target_cols}
        tag_data: dict[str, list] = {c: [] for c in tag_cols}

        for _, df_gb in grouped:
            # Convert grouped columns into arrays or scalars
            for c in feature_cols:
                vals = df_gb[c].to_numpy()
                feature_data[c].append(vals if len(vals) > 1 else vals[0])
            for c in target_cols:
                vals = df_gb[c].to_numpy()
                target_data[c].append(vals if len(vals) > 1 else vals[0])
            for c in tag_cols:
                unique_vals = df_gb[c].unique()
                tag_data[c].append(unique_vals[0] if len(unique_vals) == 1 else unique_vals.tolist())

        # 4. Ensure data is a np.ndarray for table construction
        feature_data = {k: to_numpy(v) for k, v in feature_data.items()}
        target_data = {k: to_numpy(v) for k, v in target_data.items()}
        tag_data = {k: to_numpy(v) for k, v in tag_data.items()}

        # 5. Build table
        table = build_sample_schema_table(
            features=feature_data,
            targets=target_data,
            tags=tag_data,
        )

        # 6. Return new FeatureSet
        return cls(label=str(label), table=table)

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
        Construct a FeatureSet from a Python dictionary of column data.

        Description:
            Converts a dictionary of lists/arrays into a column-oriented Arrow table
            following the ModularML `SampleSchema` convention. Each key in the input
            dictionary corresponds to a column name, and values are list-like sequences
            of equal length representing sample data.

            Unlike `from_pandas`, this constructor assumes that each list element
            already represents a complete sample (i.e., no grouping is applied).

        Args:
        label (str):
            Label to assign to this FeatureSet.
        data (dict[str, list[Any]]):
            A mapping from column names to list-like column data. Each list must
            have the same length, corresponding to the total number of samples.
        feature_keys (str | list[str]):
            Column name(s) in `data` to be used as features.
        target_keys (str | list[str]):
            Column name(s) in `data` to be used as targets.
        tag_keys (str | list[str] | None, optional):
            Column name(s) corresponding to identifying or categorical metadata
            (e.g., cell ID, protocol, SOC). Defaults to None.

        Returns:
            FeatureSet:
                A new Arrow-backed FeatureSet containing all provided columns, organized
                into standardized domains (`features`, `targets`, and `tags`).

        Raises:
            ValueError:
                If required keys are missing, or column lengths are inconsistent.

        Example:
            ```python
            fs = FeatureSet.from_dict(
                label="CycleData",
                data={
                    "voltage": [[3.1, 3.2, 3.3], [3.2, 3.3, 3.4]],
                    "current": [[1.0, 1.1, 1.0], [0.9, 1.0, 1.1]],
                    "soh": [0.95, 0.93],
                    "cell_id": ["A1", "A2"],
                },
                feature_keys=["voltage", "current"],
                target_keys=["soh"],
                tag_keys=["cell_id"],
            )
            ```

        """
        # 1. Standardize input args
        feature_keys = ensure_list(feature_keys)
        target_keys = ensure_list(target_keys)
        tag_keys = ensure_list(tag_keys)

        # 2. Ensure data is a np.ndarray for table construction
        feature_data = {k: to_numpy(data[k]) for k in feature_keys}
        target_data = {k: to_numpy(data[k]) for k in target_keys}
        tag_data = {k: to_numpy(data[k]) for k in tag_keys}

        # 3. Build table
        table = build_sample_schema_table(
            features=feature_data,
            targets=target_data,
            tags=tag_data,
        )

        # 4. Return new FeatureSet
        return cls(label=str(label), table=table)

    # ==========================================
    # GraphNode properties
    # ==========================================
    @property
    def allows_upstream_connections(self) -> bool:
        return False  # FeatureSets do not accept inputs

    @property
    def allows_downstream_connections(self) -> bool:
        return True  # FeatureSets can feed into downstream nodes

    @property
    def input_shape_spec(self) -> ShapeSpec | None:
        return None  # Not applicable

    @property
    def output_shape_spec(self) -> ShapeSpec:
        return ShapeSpec(shapes=self.column_shapes("features"))

    @property
    def max_inputs(self) -> int | None:
        return 0

    # ==========================================
    # FeatureSet Properties & Dunders
    # ==========================================
    @property
    def available_collections(self) -> list[str]:
        return list(self._collections.keys())

    @property
    def available_splits(self) -> dict[str, list[str]]:
        """
        All available splits organized by collection.

        Returns:
            dict[str, list[str]]:
                Mapping from collection key to list of split names.

        Example:
            ```python
            fs.available_splits
            # -> {"original": ["train", "val"], "transformed": ["train", "val", "test"]}
            ```

        """
        all_splits = {k: list(self._splits[k].keys()) for k in self._splits}
        return all_splits

    def get_available_splits(self, collection_key: str | None = None) -> list[str]:
        """
        Get available splits for a specific collection.

        Args:
            collection_key (str | None, optional):
                Name of the collection to query. If omitted, uses the active
                collection (`self._active_key`) but raises a warning recommending
                explicit specification.

        Returns:
            list[str]: List of split names for the specified collection.

        Raises:
            KeyError: If the collection does not exist.

        Example:
            ```python
            fs.get_available_splits("original")
            fs.get_available_splits()  # Infers active collection (and raises warning)
            ```

        """
        # Handle implicit collection access
        if collection_key is None:
            if self._active_collection is None:
                raise KeyError("No active collection set; please specify `collection_key` explicitly.")
            warnings.warn(
                f"Using active collection '{self._active_collection}' implicitly. Explicitly specify `collection_key` for clarity.",
                category=UserWarning,
                stacklevel=2,
            )
            collection_key = self._active_collection

        try:
            return list(self._splits[collection_key].keys())
        except KeyError:
            msg = f"Collection '{collection_key}' not found in FeatureSet. Available: {list(self._collections.keys())}"
            raise KeyError(msg) from None

    @property
    def n_splits(self) -> dict[str, int]:
        """
        Number of available splits per collection.

        Returns:
            dict[str, int]:
                Mapping from collection key to number of defined splits.

        Example:
            ```python
            fs.n_splits
            # -> {"original": 2, "transformed": 3}
            ```

        """
        return {coll: len(self._splits[coll]) for coll in self._splits}

    def get_n_splits(self, collection_key: str | None = None) -> int:
        """
        Get the number of splits in a specific collection.

        Args:
            collection_key (str | None, optional):
                Name of the collection to query. If omitted, uses the active
                collection (`self._active_key`) but raises a warning recommending
                explicit specification.

        Returns:
            int: Number of splits in the specified collection.

        Raises:
            KeyError: If the collection does not exist.

        Example:
            ```python
            fs.get_n_splits("original")
            fs.get_n_splits()  # Infers active collection (and raises warning)
            ```

        """
        # Handle implicit collection access
        if collection_key is None:
            active = getattr(self, "_active_key", None)
            if active is None:
                raise KeyError("No active collection set; please specify `collection_key` explicitly.")
            warnings.warn(
                f"Using active collection '{active}' implicitly. Explicitly specify `collection_key` for clarity.",
                category=UserWarning,
                stacklevel=2,
            )
            collection_key = active

        try:
            return len(self._splits[collection_key])
        except KeyError:
            msg = f"Collection '{collection_key}' not found in FeatureSet. Available: {list(self._collections.keys())}"
            raise KeyError(msg) from None

    def __getitem__(self, key: str) -> FeatureSetView:
        """Alias for get_split(key)."""
        return self.get_split(key)

    def __len__(self) -> int:
        """Returns number of samples in this FeatureSet."""
        return self._collections[self._default_collection_key].n_samples

    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"

    # ==========================================
    # SampleCollection Accessors
    # ==========================================
    @property
    def original(self) -> SampleCollection:
        return self._collections[self._default_collection_key]

    @property
    def transformed(self) -> SampleCollection:
        return self._collections[self._transformed_collection_key]

    @property
    def active(self) -> SampleCollection:
        return self._collections[self._active_key]

    def set_active(self, collection_key: str) -> None:
        if collection_key not in self._collections:
            msg = f"No SampleCollection named '{collection_key}'"
            raise KeyError(msg)
        self._active_key = collection_key

    def get_collection(self, collection_key: str) -> SampleCollection:
        if collection_key not in self._collections:
            msg = f"No SampleCollection named '{collection_key}'"
            raise KeyError(msg)
        return self._collections[collection_key]

    # ==========================================
    # Split (FeatureSetView) Utilities
    # ==========================================
    def get_split(
        self,
        key: str | None = None,
        *,
        collection_key: str | None = None,
        split_name: str | None = None,
    ) -> FeatureSetView:
        """
        Retrieve a split (FeatureSetView) by name or namespace.

        Description:
            Provides flexible lookup for FeatureSet splits, supporting several
            access styles while maintaining clear semantics:
            1. Explicit (recommended)
                `get_split(collection_key="original", split_name="train")`
                Retrieves the "train" split from the "original" collection.
            2. Namespaced key
                `get_split("original.train")`
                Automatically parses the dotted key into ("original", "train").
            3. Implicit active collection
                `get_split("train")`
                Uses the active collection key (`self._active_key`) implicitly.
                This is allowed but raises a warning recommending explicit usage.

        Args:
            key (str | None, optional):
                A single key string. May be a dotted name ("original.train")
                or a bare split name ("train"), in which case the active collection
                is assumed.
            collection_key (str | None, optional):
                Name of the collection to query (e.g., "original").
                Must be provided if `key` is not given.
            split_name (str | None, optional):
                Name of the split (e.g., "train"). Must be provided when using
                explicit access mode.

        Returns:
            FeatureSetView:
                The requested FeatureSetView.

        Raises:
            ValueError:
                If argument combinations are invalid or ambiguous.
            KeyError:
                If the requested split or collection does not exist.

        Example:
            ```python
            fs.get_split(collection_key="original", split_name="train")  # recommended
            fs.get_split("original.train")  # shorthand
            fs.get_split("train")  # infers active collection (warning raised)
            ```

        """
        # Validate argument combinations
        provided = [arg is not None for arg in (key, collection_key, split_name)]
        if sum(provided) == 0:
            raise ValueError("Must provide either `key` or (`collection_key` + `split_name`).")
        if key and (collection_key or split_name):
            raise ValueError("Cannot mix positional `key` with explicit `collection_key` or `split_name`.")

        # Case 1: single composite or bare key
        if key is not None:
            if "." in key:  # e.g. "original.train"
                coll, split = key.split(".", 1)
            else:
                coll = self._active_collection
                if coll is None:
                    msg = f"No active collection defined. Explicitly specify collection_key when calling get_split('{key}')."
                    raise ValueError(msg)
                warnings.warn(
                    f"Inferred active collection '{coll}' for split '{key}'. "
                    "This is allowed but explicit specification is recommended.",
                    UserWarning,
                    stacklevel=2,
                )
                split = key
        else:
            # Case 2: fully explicit
            if not (collection_key and split_name):
                raise ValueError("Must provide both `collection_key` and `split_name` for explicit access.")
            coll, split = collection_key, split_name

        # Lookup
        try:
            return self._splits[coll][split]
        except KeyError:
            msg = f"Split '{split}' not found in collection '{coll}'. Available: {self.available_splits}"
            raise KeyError(msg) from None

    def add_split(self, split: FeatureSetView):
        """
        Adds a new FeatureSetView.

        Args:
            split (FeatureSetView): The new view to add.

        """
        # Check that split references this instance and collection exists
        if split.source is not self:
            msg = f"New split `{split.label}` is not a view of this FeatureSet instance."
            raise ValueError(msg)
        if split.collection_key not in self._collections:
            msg = (
                f"New split `{split.label}` references a collection that does not exist in this FeatureSet: "
                f"`{split.collection_key}`."
            )
            raise ValueError(msg)

        # Check that split name is unique
        if split.collection_key not in self._splits:
            self._splits[split.collection_key] = {}
        used_split_names = self.get_available_splits(collection_key=split.collection_key)
        if split.label is None or split.label in used_split_names:
            msg = f"Split label ('{split.label}') is missing or already exists."
            raise ValueError(msg)

        # Check that new split name follows naming conventions
        try:
            validate_str_list(used_split_names + split.label)
        except ValueError as e:
            msg = f"Failed to add new split `{split.label}`. {e}"
            raise RuntimeError(msg) from e

        # Check overlap with existing splits (only within the same collection)
        for existing_split in self._splits[split.collection_key].values():
            if not split.is_disjoint_with(existing_split):
                overlap: list[int] = split.get_overlap_with(existing_split)
                warnings.warn(
                    f"\nSplit '{split.label}' has overlapping samples with existing split '{existing_split.label}'.\n"
                    f"    (n_overlap = {len(overlap)})\n"
                    f"    Consider checking for disjoint split or revising your conditions.",
                    SplitOverlapWarning,
                    stacklevel=2,
                )

        # Register new split
        self._splits[split.collection_key][split.label] = split

        # Update active collection
        self.set_active(split.collection_key)

    def clear_splits(self, collection_key: str | None = None) -> None:
        """
        Removes all previously defined splits.

        Args:
            collection_key (str | None, optional):
                If provided, clears splits only for that collection.
                If None, clears all splits across all collections.

        """
        if collection_key is None:
            self._splits.clear()
        else:
            self._splits.pop(collection_key, None)

    # ================================================================================
    # Splitting Methods
    # ================================================================================
    def filter(
        self,
        *,
        collection_key: str | None = None,
        **conditions: dict[str, Any | list[Any], Callable],
    ) -> FeatureSetView | None:
        """
        Create a filtered view over a specific collection based on tag, feature, or target conditions.

        Description:
            Filters the underlying Arrow table according to key-value conditions \
            applied across all schema domains ("features", "targets", "tags"). \
            Each condition may be:
                - A literal value for equality matching.
                - A list/tuple/set/np.ndarray of allowed values.
                - A callable that takes a NumPy or Arrow array and returns a boolean mask.

            Returns a lightweight :class:`FeatureSetView` referencing the filtered rows \
            of the current FeatureSet (without copying underlying data).

        Args:
            collection_key (str | None, optional):
                The collection to filter. If None, uses the inferred active collection.
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

        # Resolve target collection
        if collection_key is None:
            if self._active_collection is None:
                raise KeyError("No active collection set; please specify `collection_key`.")
            collection_key = self._active_collection

        # Get mask (np.ndarry of collection indices) using conditions
        mask = _evaluate_filter_conditions(
            collection=self._collections[collection_key],
            conditions=conditions,
        )

        # Convert mask to indices
        selected_indices = np.where(mask)[0]
        if len(selected_indices) == 0:
            warnings.warn(
                f"No samples match filter conditions: {list(conditions.keys())}",
                UserWarning,
                stacklevel=2,
            )

        # Build FeatureSetView using indices
        return FeatureSetView(
            source=self,
            collection_key=collection_key,
            indices=selected_indices,
            label="filtered",
        )

    def _as_view(self, collection_key: str | None = None) -> FeatureSetView:
        """
        Create a full FeatureSetView over a specific collection.

        Args:
            collection_key (str | None, optional):
                The name of the collection to view.
                If None, uses the inferred active collection.

        Returns:
            FeatureSetView: A view referencing all rows of the given collection.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        if collection_key is None:
            if self._active_collection is None:
                raise KeyError("No active collection set; please specify `collection_key`.")
            collection_key = self._active_collection

        coll = self._collections[collection_key]
        return FeatureSetView(
            source=self,
            collection_key=collection_key,
            indices=np.arange(coll.n_samples),
            label=f"{collection_key}_view",
        )

    def split(
        self,
        splitter: BaseSplitter,
        *,
        collection_key: str | None = None,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Apply a splitter to this FeatureSet (or one of its collections).

        Description:
            Runs the provided `BaseSplitter` instance on a view of the specified \
            collection (e.g., 'original' or 'transformed'). The resulting splits \
            (and splitter config) are optionally registered into the FeatureSet's \
            `_splits` registry.

        Args:
            splitter (BaseSplitter):
                The splitter instance (e.g., RandomSplitter, ConditionSplitter).
            collection_key (str | None, optional):
                Which collection to split. Defaults to the current active collection.
            return_views (bool, optional):
                Whether to return FeatureSetViews or not. Defaults to False.
            register (bool, optional):
                Whether to record the resulting views and splitter config.
                Defaults to True.

        Returns:
            list[FeatureSetView] | None:
                The created splits are returned only if `return_views=True`.

        Example:
            ```python
            splitter = RandomSplitter({"train": 0.8, "val": 0.2}, group_by="cell_id")
            fs.split(splitter, collection_key="original")
            ```

        """
        # Determine which collection to split
        if collection_key is None:
            collection_key = self._active_collection
        if collection_key not in self._collections:
            msg = f"Collection '{collection_key}' not found. Available: {list(self._collections.keys())}"
            raise KeyError(msg)

        # Create a view over the collection
        base_view = self._as_view(collection_key=collection_key)

        # Perform the split (return splits as FeatureSetView instances)
        results: dict[str, FeatureSetView] = splitter.split(base_view, return_views=True)
        results = list(results.values())

        # Register splits if requested
        if register:
            for split in results:
                self.add_split(split)

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
        collection_key: str | None = None,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Randomly partition this FeatureSet (or one of its collections) into subsets.

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
            collection_key (str | None, optional):
                Name of the collection to split (e.g., `"original"` or `"transformed"`).
                Defaults to the currently active collection.
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
                collection_key="original",
            )
            ```

        """
        from modularml.core.pipeline.splitting.random_splitter import RandomSplitter

        splitter = RandomSplitter(ratios, group_by=group_by, seed=seed)
        return self.split(splitter, collection_key=collection_key, return_views=return_views, register=register)

    def split_by_condition(
        self,
        conditions: dict[str, dict[str, Any]],
        *,
        collection_key: str | None = None,
        return_views: bool = False,
        register: bool = True,
    ) -> list[FeatureSetView] | None:
        """
        Split this FeatureSet (or one of its collections) based on logical conditions.

        Description:
            A convenience wrapper around :class:`ConditionSplitter`, which partitions
            samples into subsets based on user-defined filter expressions.

            Each subset is defined by a dictionary mapping feature, target, or tag keys
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

            **Note:** Overlapping subsets are permitted, but a warning will be issued
            if any sample satisfies multiple conditions.

        Args:
            conditions (Mapping[str, Mapping[str, Any | Sequence | Callable]]):
                Mapping of subset labels → condition dictionaries.
                Each condition dictionary maps a key (feature/target/tag name)
                to a condition (scalar, sequence, or callable).
            collection_key (str | None, optional):
                Name of the collection to split (e.g., `"original"` or `"transformed"`).
                Defaults to the currently active collection.
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
            fs.split_by_condition(
                {
                    "train": {"cell_type": "A"},
                    "test": {"cell_type": "B"},
                }
            )
            ```

        """
        from modularml.core.pipeline.splitting.condition_splitter import ConditionSplitter

        splitter = ConditionSplitter(conditions=conditions)
        return self.split(splitter, collection_key=collection_key, return_views=return_views, register=register)


# TODO: TASKS:
# - Splitting methods implemented into FeatureSet (not FeatureSetView)
#   - Need to implement into FeatureSetView
#   - Also need to track splitting history / config

# - Need to implement FeatureTransform logic
#   - How to track transform history
