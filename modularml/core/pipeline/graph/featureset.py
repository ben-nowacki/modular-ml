from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa

# from modularml.core.graph.feature_subset import FeatureSubset
# from modularml.core.transforms.feature_transform import FeatureTransform
from modularml.components.graph_node import GraphNode, ShapeSpec
from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_schema import FEATURES_COLUMN, SAMPLE_ID_COLUMN, TAGS_COLUMN, TARGETS_COLUMN
from modularml.utils.data_conversion import to_numpy
from modularml.utils.data_format import DataFormat, ensure_list
from modularml.utils.exceptions import SampleLoadError, SplitOverlapWarning
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
class FeatureSet(SampleCollection, GraphNode):
    """
    Arrow-backed FeatureSet wrapping a SampleCollection.

    Acts as the first data node in a ModularML pipeline. Contains data,
    subset definitions, and transformation/splitting logic.
    """

    def __init__(
        self,
        label: str,
        table: pa.Table,
        schema: SampleSchema | None = None,
    ):
        SampleCollection.__init__(self, table=table, schema=schema)
        GraphNode.__init__(self, label=label)

        self._splits: dict[str, FeatureSetView] = {}
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
    def available_splits(self) -> list[str]:
        return list(self._splits.keys())

    @property
    def splits(self) -> dict[str, FeatureSetView]:
        return self._splits

    @property
    def n_splits(self) -> int:
        return len(self.available_splits)

    def __len__(self) -> int:
        """Returns number of samples in this FeatureSet."""
        return self.n_samples

    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"

    # ==========================================
    # Split (FeatureSetView) Utilities
    # ==========================================
    def get_split(self, name: str) -> FeatureSetView:
        """
        Returns the specified split (a FeatureSetView) of this FeatureSet.

        Use `FeatureSet.available_splits` to view available split names.

        Args:
            name (str): Split name to return.

        Returns:
            FeatureSetView: A named view of this FeatureSet.

        """
        if name not in self._splits:
            msg = (
                f"`{name}` is not a valid split of {self.label}. "
                f"Use `FeatureSet.available_splits` to view available split names."
            )
            raise ValueError(msg)

        return self._splits[name]

    def add_split(self, split: FeatureSetView):
        """
        Adds a new FeatureSetView.

        Args:
            split (FeatureSetView): The new view to add.

        """
        # Check that new split name is unique
        if split.label is None or split.label in self._splits:
            msg = f"Split label ('{split.label}') is missing or already exists."
            raise ValueError(msg)

        # Check that new split is a view of this FeatureSet
        if split.source is not self:
            msg = "Split must be a view of this FeatureSet."
            raise ValueError(msg)

        # Check overlap with existing splits
        for sub in self._splits.values():
            if not split.is_disjoint_with(sub):
                overlap: list[int] = split.get_overlap_with(sub)
                warnings.warn(
                    f"\nSplit '{split.label}' has overlapping samples with existing split '{sub.label}'.\n"
                    f"    (n_overlap = {len(overlap)})\n"
                    f"    Consider checking for disjoint split or revising your conditions.",
                    SplitOverlapWarning,
                    stacklevel=2,
                )

        # Add to internal list of splits
        self._splits[split.label] = split

    def clear_splits(self) -> None:
        """Remove all previously defined splits."""
        self._splits.clear()
        # self._split_configs = []

    # ================================================================================
    # Splitting Methods
    # ================================================================================
    def filter(
        self,
        **conditions: dict[str, Any | list[Any], Callable],
    ) -> FeatureSetView | None:
        """
        Create a filtered view of this FeatureSet based on tag, feature, or target conditions.

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

        # Start with all rows included
        mask = np.ones(self.n_samples, dtype=bool)

        # Check available keys
        # We want to search this in tag -> feature -> target order
        ordered_domains = [
            (TAGS_COLUMN, self.tag_keys),
            (FEATURES_COLUMN, self.feature_keys),
            (TARGETS_COLUMN, self.target_keys),
        ]
        all_domains = OrderedDict(ordered_domains)

        for key, cond in conditions.items():
            # Find pyarrow domain that matches this key (use first match)
            domain_of_key = next(d for d, d_keys in all_domains.items() if key in d_keys)
            if domain_of_key is None:
                msg = (
                    f"Key '{key}' not found in features, targets, or tags. "
                    f"Use `.tag_keys`, `.feature_keys`, or `.target_keys` to see all available keys."
                )
                raise KeyError(msg)

            # Filter pyarrow table to column specified by 'key'
            col_data = self._domain_dataframe(domain=domain_of_key, keys=[key]).to_numpy()

            # Evaluate condition
            if callable(cond):
                try:
                    local_mask = np.asarray(cond(col_data))
                except Exception as e:
                    msg = f"Failed to apply callable conditon for key '{key}': {e}"
                    raise ValueError(msg) from e
            elif isinstance(cond, list | tuple | set | np.ndarray):
                local_mask = np.isin(col_data, cond, assume_unique=False)
            else:
                local_mask = col_data == cond

            # Combine with global mask
            local_mask = local_mask.reshape(self.n_samples)
            mask &= local_mask

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
            indices=selected_indices,
            label="filtered",
        )

    def _as_view(self) -> FeatureSetView:
        """Creates a view over the entire FeatureSet."""
        from modularml.core.data.featureset_view import FeatureSetView

        return FeatureSetView(
            source=self,
            indices=np.arange(self.n_samples),
            label=self.label,
        )


# TASKS:
# - Splitters can be run directly, but not impemented into FeatureSet yet.
#   - Need to track splitting history and available splits

# - No support for feature transforms
#   - Do we use a second copy of SampleCollection? How when inherits?
