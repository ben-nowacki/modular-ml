from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from modularml.components.graph_node import GraphNode, ShapeSpec
from modularml.core.data.sample_collection import SampleCollection
from modularml.core.data.sample_schema import (
    FEATURES_COLUMN,
    RAW_VARIANT,
    TAGS_COLUMN,
    TARGETS_COLUMN,
    TRANSFORMED_VARIANT,
    validate_str_list,
)
from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.core.splitting.split_mixin import SplitMixin, SplitterRecord
from modularml.core.transforms.scaler import Scaler
from modularml.core.transforms.scaler_record import ScalerRecord
from modularml.utils.data_conversion import flatten_to_2d, to_numpy, unflatten_from_2d
from modularml.utils.data_format import DataFormat, ensure_list
from modularml.utils.exceptions import SplitOverlapWarning
from modularml.utils.pyarrow_data import build_sample_schema_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa

    from modularml.core.data.featureset_view import FeatureSetView
    from modularml.core.data.sample_schema import SampleSchema


class FeatureSet(GraphNode, SplitMixin):
    """
    Unified FeatureSet backed by a single SampleCollection.

    Each variant (e.g., "raw", "transformed") lives within the same \
    SampleCollection rather than as separate sub-collections. \
    Splits store indices into this collection and may specify \
    which variant(s) to use when retrieving data.
    """

    def __init__(
        self,
        label: str,
        table: pa.Table,
        schema: SampleSchema | None = None,
    ):
        # Instaniate GraphNode with label
        super().__init__(label=label)

        # Create SampleCollection attribute
        self.collection: SampleCollection = SampleCollection(table=table, schema=schema)

        # Store splits & spliiter configs
        self._splits: dict[str, FeatureSetView] = {}
        self._split_configs: list[SplitterRecord] = []

        # Store scaler logs/configs
        self._scaler_logs: list[ScalerRecord] = []

    @classmethod
    def from_dict(
        cls,
        label: str,
        data: dict[str, Sequence[Any]],
        feature_keys: str | list[str],
        target_keys: str | list[str],
        tag_keys: str | list[str] | None = None,
    ) -> FeatureSet:
        """
        Construct a FeatureSet from a Python dictionary of column data.

        Description:
            Converts a dictionary of lists/arrays into a column-oriented Arrow table \
            following the ModularML `SampleSchema` convention. Each key in the input \
            dictionary corresponds to a column name, and values are list-like sequences \
            of equal length representing sample data.

            Unlike `from_pandas`, this constructor assumes that each list element \
            already represents a complete sample (i.e., no grouping is applied).

        Args:
        label (str):
            Label to assign to this FeatureSet.
        data (dict[str, Sequence[Any]]):
            A mapping from column names to list-like column data. Each list must \
            have the same length, corresponding to the total number of samples.
        feature_keys (str | list[str]):
            Column name(s) in `data` to be used as features.
        target_keys (str | list[str]):
            Column name(s) in `data` to be used as targets.
        tag_keys (str | list[str] | None, optional):
            Column name(s) corresponding to identifying or categorical metadata \
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

    def __eq__(self, other):
        if not isinstance(other, FeatureSet):
            msg = f"Cannot compare equality between FeatureSet and {type(other)}"
            raise TypeError(msg)
        return (
            self.label == other.label
            and self.collection == other.collection
            # and self._splits == other._splits  # don't need to comapre views, just configs
            and self._split_configs == other._split_configs
            and self._scaler_logs == other._scaler_logs
        )

    __hash__ = None

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
        return ShapeSpec(
            shapes=self.collection.feature_shapes,
            num_samples=self.collection.n_samples,
        )

    @property
    def max_inputs(self) -> int | None:
        return 0

    # ==========================================
    # SampleCollection properties
    # ==========================================
    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self.collection.n_samples

    @property
    def feature_keys(self) -> list[str]:
        """List of feature column names."""
        return self.collection.feature_keys

    @property
    def target_keys(self) -> list[str]:
        """List of target column names."""
        return self.collection.target_keys

    @property
    def tag_keys(self) -> list[str]:
        """List of tag column names."""
        return self.collection.tag_keys

    @property
    def feature_shapes(self) -> ShapeSpec:
        """
        Mapping of feature columns (including variants) to their data shapes.

        Description:
            Returns a ShapeSpec instance containing per feature shapes and a separate \
            number of sample property. See `ShapeSpec` for more details.

        Returns:
            ShapeSpec:
                Per-feature shapes.

        """
        return ShapeSpec(
            shapes=self.collection.feature_shapes,
            num_samples=self.collection.n_samples,
        )

    @property
    def target_shapes(self) -> ShapeSpec:
        """
        Mapping of target columns (including variants) to their data shapes.

        Description:
            Returns a ShapeSpec instance containing per target shapes and a separate \
            number of sample property. See `ShapeSpec` for more details.

        Returns:
            ShapeSpec:
                Per-target shapes.

        """
        return ShapeSpec(
            shapes=self.collection.target_shapes,
            num_samples=self.collection.n_samples,
        )

    @property
    def tag_shapes(self) -> ShapeSpec:
        """
        Mapping of tag columns (including variants) to their data shapes.

        Description:
            Returns a ShapeSpec instance containing per tag shapes and a separate \
            number of sample property. See `ShapeSpec` for more details.

        Returns:
            ShapeSpec:
                Per-tag shapes.

        """
        return ShapeSpec(
            shapes=self.collection.tag_shapes,
            num_samples=self.collection.n_samples,
        )

    @property
    def feature_dtypes(self) -> dict[str, str]:
        """
        Mapping of feature columns (including variants) to their data types.

        Description:
            Returns a dictionary describing the data type of each feature variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of feature column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "voltage.raw": "float32",
                "voltage.transformed": "float32",
            }
        ```

        """
        return self.collection.feature_dtypes

    @property
    def target_dtypes(self) -> dict[str, str]:
        """
        Mapping of target columns (including variants) to their data types.

        Description:
            Returns a dictionary describing the data type of each target variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of target column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "soh.raw": "float32",
                "soh.transformed": "float32",
            }
        ```

        """
        return self.collection.target_dtypes

    @property
    def tag_dtypes(self) -> dict[str, str]:
        """
        Mapping of tag columns (including variants) to their data types.

        Description:
            Returns a dictionary describing the data type of each tag variant \
            in the FeatureSet. Keys are formatted as `"column.variant"` and \
            data types are strings (e.g., "float32").

        Returns:
            dict[str, str]:
                Mapping of tag column variant names to their data types (as strings).

        Example:
        ``` python
            {
                "cell_id.raw": "string",
                "cell_id.transformed": "int8",
            }
        ```

        """
        return self.collection.tag_dtypes

    def get_features(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve feature data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of feature keys to return. \
                If None, all feature keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the feature keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                feature keys (e.g., "voltage" or "voltage.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                feature keys (e.g., "voltage" or "features.voltage"). Defaults to False.

        Returns:
            Feature data in the requested format.

        """
        return self.collection.get_features(
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_targets(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve target data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of target keys to return. \
                If None, all target keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the target keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                target keys (e.g., "soh" or "soh.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                target keys (e.g., "soh" or "targets.soh"). Defaults to False.

        Returns:
            Target data in the requested format.

        """
        return self.collection.get_targets(
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    def get_tags(
        self,
        fmt: DataFormat = DataFormat.DICT_NUMPY,
        *,
        keys: str | list[str] | None = None,
        variant: str | None = RAW_VARIANT,
        include_variant_suffix: bool = False,
        include_domain_prefix: bool = False,
    ):
        """
        Retrieve tag data in a chosen format.

        Args:
            fmt (DataFormat): Desired output format (see :class:`DataFormat`). \
                Defaults to a single dictionary of numpy arrays.
            keys (str | list[str] | None): Optional subset of tag keys to return. \
                If None, all tag keys are returned. Defaults to None.
            variant (str, optional): The variant (e.g., "raw" or "transformed") of the tag keys to \
                return. If None, all variants are returned and `include_variant_suffix` is set to True. \
                If specfied, `keys` must all have matching variants. Defaults to RAW_VARIANT.
            include_variant_suffix (bool): Whether to include the variant suffix in the \
                tag keys (e.g., "cell_id" or "cell_id.raw"). Defaults to False.
            include_domain_prefix (bool): Wether to include the domain prefix in the \
                tag keys (e.g., "cell_id" or "tags.cell_id"). Defaults to False.

        Returns:
            Tag data in the requested format.

        """
        return self.collection.get_tags(
            fmt=fmt,
            keys=keys,
            variant=variant,
            include_variant_suffix=include_variant_suffix,
            include_domain_prefix=include_domain_prefix,
        )

    # ==========================================
    # FeatureSet Properties & Dunders
    # ==========================================
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, key: str) -> FeatureSetView:
        """Alias for get_split(key)."""
        return self.get_split(key)

    def __repr__(self):
        return f"FeatureSet(label='{self.label}', n_samples={len(self)})"

    def __str__(self):
        return self.__repr__()

    def _as_view(self) -> FeatureSetView:
        """
        Create a FeatureSetView over the entire FeatureSet.

        Returns:
            FeatureSetView: A view referencing all rows.

        """
        from modularml.core.data.featureset_view import FeatureSetView

        return FeatureSetView(
            source=self,
            indices=np.arange(self.collection.n_samples),
            label=f"{self.label}_view",
        )

    @property
    def splits(self) -> dict[str, FeatureSetView]:
        return self._splits

    @property
    def n_splits(self) -> int:
        return len(self._splits)

    # ==========================================
    # Split Utilities
    # ==========================================
    # Most splitting logic is handled in the SplitterMixin class
    @property
    def available_splits(self) -> list[str]:
        """
        All available splits.

        Returns:
            list[str]: Available split names.

        """
        return list(self._splits.keys())

    def get_split(self, split_name: str) -> FeatureSetView:
        """
        Gets the specified split.

        Returns:
            FeatureSetView:
                A no-copy, filtered view of the FeatureSet.

        """
        if split_name not in self._splits:
            msg = f"Split '{split_name}' does not exist. Available: {self.available_splits}"
            raise KeyError(msg)
        return self._splits[split_name]

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

        # Check that split name is unique
        if split.label is None or split.label in self.available_splits:
            msg = f"Split label ('{split.label}') is missing or already exists."
            raise ValueError(msg)

        # Check that new split name follows naming conventions
        try:
            validate_str_list([*self.available_splits, split.label])
        except ValueError as e:
            msg = f"Failed to add new split `{split.label}`. {e}"
            raise RuntimeError(msg) from e

        # Check overlap with existing splits (only within the same collection)
        for existing_split in self._splits.values():
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
        self._splits[split.label] = split

    def clear_splits(self) -> None:
        """Removes all previously defined splits."""
        self._splits.clear()
        self._split_configs.clear()

    # ==========================================
    # Transform/Scaling
    # ==========================================
    def _get_flat_data(
        self,
        domain: str,
        keys: list[str],
        variant: str,
        split: str | None,
        merged_axes: int | tuple[int],
    ) -> tuple[np.ndarray, dict]:
        """Returns new 2D array and metadata of flattening process."""
        # Get collection to flatten
        coll = self.collection if split is None else self.get_split(split).to_samplecollection()

        # Get data specified by domain + keys + variant
        data = coll.get_domain_data(
            domain=domain,
            fmt=DataFormat.NUMPY,
            keys=keys,
            variant=variant,
        )

        # Scaler requires 2D array
        # If we need to reshape, check `merged_axes`
        #   - If None, throw error that shapes are >2D, print shapes and state valid
        #     indices to use for merged_axes
        #   - If an integer, preserve the specified axis, and flatten the other(s)
        flat_data, flat_meta = data, {}
        if data.ndim > 2:
            if merged_axes is None:
                # Common use case is 1D data with 1 feature key (eg, shape = (n_sample, 1, n_f))
                # We can just reshape to (n_sample, n_f) without throwing error
                if data.ndim == 3 and len(keys) == 1:
                    merged_axes = (1, 2)
                else:
                    msg = (
                        f"Scalers can only be fit to 2-dimensional data, `{domain}.{keys}` "
                        f"results in data with shape: {data.shape}. "
                        f"Use `merged_axes` to combine the specified axes and flatten the rest."
                    )
                    raise ValueError(msg)
            flat_data, flat_meta = flatten_to_2d(arr=data, merged_axes=merged_axes)

        if data.ndim < 2:
            # Add extra dimension to make 2D
            flat_data, flat_meta = flatten_to_2d(arr=data, merged_axes=0)

        return flat_data, flat_meta

    def fit_transform(
        self,
        scaler: Scaler | Any,
        domain: str,
        keys: str | list[str] | None = None,
        fit_to_split: str | None = None,
        merged_axes: int | tuple[int] | None = None,
    ):
        """
        Fit a scaler to selected columns and apply the transform to the entire FeatureSet.

        Description:
            - Flattens the selected columns (2D), optionally merging axes (`merged_axes`).
            - Fits the provided scaler to samples from `fit_to_split` (or all samples).
            - Transforms all samples.
            - Unflattens results back to original shapes.
            - Stores a new variant (`TRANSFORMED_VARIANT`) for each key.
            - Appends a ScalerRecord to the transform log.

        Args:
            scaler (Scaler | Any):
                A Scaler instance or any sklearn-like object implementing `fit()` and \
                `transform()`. If not a Scaler, it is wrapped as `Scaler(scaler=<obj>)`.
            domain (str):
                One of {"features", "targets", "tags"}.
            keys (str | list[str] | None, optional):
                Column names within the domain to transform. If None, all domain keys \
                are used. Defaults to None.
            fit_to_split (str | None, optional):
                Split name from which to fit the scaler (e.g., "train"). If None, the \
                scaler is fit to all samples. Defaults to None.
            merged_axes (int | tuple[int]):
                Axes whose sizes are merged into a single dimension. If None, no axes are \
                merged. An Error will be thrown if the resulting shape is not 2-dimensional.

        Raises:
            ValueError:
                - If flattening with `merged_axes` does not result in 2D data.
                - If variant or domain keys do not exist.
            TypeError:
                If `scaler` is not a valid Scaler or sklearn-transformer-like object.

        """
        # Ensure scaler is a Scaler instance
        if not isinstance(scaler, Scaler):
            scaler = Scaler(scaler=scaler)

        # Infer variant to fit to:
        #   - if all specified keys already have a "transformed" variant, use that
        #   - otherwise, use the "raw" variant

        # Use specified keys, or all keys
        keys = ensure_list(keys)
        if not keys:
            keys = self.collection.get_domain_keys(domain=domain)

        # Find common variant
        variant_to_use = RAW_VARIANT
        if self._scaler_logs:
            has_transformed_variant = True
            for k in keys:
                existing_vars = self.collection.get_variant_keys(domain=domain, key=k)
                if TRANSFORMED_VARIANT not in existing_vars:
                    has_transformed_variant = False
                    break
            variant_to_use = TRANSFORMED_VARIANT if has_transformed_variant else RAW_VARIANT

        # Get data to fit to and fit Scaler
        x_fit, _ = self._get_flat_data(
            domain=domain,
            keys=keys,
            variant=variant_to_use,
            split=fit_to_split,
            merged_axes=merged_axes,
        )
        scaler.fit(x_fit)

        # Get data to transform
        x_all, x_all_meta = self._get_flat_data(
            domain=domain,
            keys=keys,
            variant=variant_to_use,
            split=None,
            merged_axes=merged_axes,
        )
        x_trans_flat = scaler.transform(x_all)

        # Unpack transformed data back to original shapes
        x_trans = unflatten_from_2d(flat=x_trans_flat, meta=x_all_meta) if x_all_meta else x_trans_flat

        # Store this data back into collection table
        for k_idx, k in enumerate(keys):
            # x_trans has shape like (n_samples, n_f, f_shape) where n_f is number of keys
            # We need to select data belonging to each key -> shape = (n_samples, f_shape)
            self.collection.add_variant(
                domain=domain,
                key=k,
                variant=TRANSFORMED_VARIANT,
                data=x_trans[:, k_idx],
                overwrite=True,
            )

        # Record this scaler configuration
        self._scaler_logs.append(
            ScalerRecord(
                order=len(self._scaler_logs),
                domain=domain,
                keys=keys,
                variant_in=variant_to_use,
                variant_out=TRANSFORMED_VARIANT,
                fit_split=fit_to_split,
                merged_axes=merged_axes,
                flatten_meta=x_all_meta,
                scaler_object=copy.deepcopy(scaler),
            ),
        )

    def undo_last_transform(
        self,
        domain: str,
        keys: str | list[str] | None = None,
    ):
        """
        Undo the most recent transform applied to a specific subset of columns.

        Description:
            Reverts only the last ScalerRecord that:
                - matches exactly the provided `domain` + `keys`, AND
                - is not depended on by a more recent transforms.

            The method:
                - Finds the most recent exact match (same set of keys).
                - Ensures no later transform includes these keys (dependency safety).
                - Flattens data using stored metadata.
                - Applies `inverse_transform()`.
                - Unflattens back to the original shape.
                - Replaces the transformed variant.
                - Removes the ScalerRecord and updates order indices.

        Args:
            domain (str):
                One of {"features", "targets", "tags"}.
            keys (str | list[str] | None):
                Keys to revert. If None, all domain keys are used.

        Raises:
            ValueError:
                - If no transforms were applied on these columns.
                - If the last applicable transform cannot be undone due to \
                dependency on a more recent transform.
                - If flatten metadata does not match stored ScalerRecord metadata.
                - If internal transform history becomes inconsistent.

        """
        # Undoing the last transform applied to some subset of FeatureSet
        #   1. Define the subset of interest
        #   - Transform can only be undone if the specified subset of interest covers
        #       the full subset of the last transform and no other transforms overlap
        #       with this subset of interest
        #   - Example 1: successful
        #       apply X to "features.voltage"
        #       apply Y to "features.voltage"
        #       undo last on "features.voltage" --> inverts Y
        #   - Example 2: invalid
        #       apply X to "features.[voltage, current]"
        #       apply Y to "features.voltage"
        #       undo last on "features.[voltage, current]"
        #           -> the last transform on the specified domain is X
        #           -> but transform Y depends on X so we cannot invert
        #           -> must first undo Y on "features.voltage"

        # Use specified keys, or all keys
        keys = ensure_list(keys)
        if not keys:
            keys = self.collection.get_domain_keys(domain=domain)

        # Get most recent ScalerRecord for:
        #   a. scalers using only the specified domain + keys (`last_exact`)
        #   b. scalers including the specified domain + keys (`last_incl`)
        # If (a) is not more recent than or equal to (b), the transform cannot be undone
        last_exact, last_incl = None, None
        for record in reversed(self._scaler_logs):
            if record.domain == domain:
                # If record matches specified domain + keys exactly (order doesn't matter)
                if last_exact is None and set(record.keys) == set(keys):
                    last_exact = record
                # If record contains at least the specified domain + keys
                if last_incl is None and all(ki in record.keys for ki in keys):
                    last_incl = record

            if last_exact is not None and last_incl is not None:
                break
        if last_exact is None and last_incl is not None:
            msg = (
                f"There are no Scalers applied to exactly {domain}.{keys}. "
                f"There are Scalers applied that include these keys (along with other). "
                f"To undo those, use the following keys: {last_incl.keys}"
            )
            raise ValueError(msg)
        if last_exact is None and last_incl is None:
            msg = "There are no Scalers to undo."
            raise ValueError(msg)

        # Raise error if cannot undo scaler on specified domain
        if last_exact.order < last_incl.order:
            msg = (
                f"The last scaler applied to {domain}.{keys} cannot be undone as a "
                "more recent scaler depends on it. You need to first undo the scaler "
                f"applied to {last_incl.domain}.{last_incl.keys}."
            )
            raise ValueError(msg)

        # Get flattened data to inverse
        x_flat, x_flat_meta = self._get_flat_data(
            domain=last_exact.domain,
            keys=last_exact.keys,
            variant=last_exact.variant_out,
            split=None,
            merged_axes=last_exact.merged_axes,
        )
        if x_flat_meta != last_exact.flatten_meta:
            msg = (
                "Flattened data does not match expected metadata from ScalerRecord: "
                f"{x_flat_meta} != {last_exact.flatten_meta}"
            )
            raise ValueError(msg)

        # Inverse data and reshape
        x_inv_flat = last_exact.scaler_object.inverse_transform(x_flat)
        x_inv = unflatten_from_2d(flat=x_inv_flat, meta=x_flat_meta)

        # Store this data back into collection table
        for k_idx, k in enumerate(last_exact.keys):
            self.collection.add_variant(
                domain=last_exact.domain,
                key=k,
                variant=last_exact.variant_out,
                data=x_inv[:, k_idx],
                overwrite=True,
            )

        # Delete record from logs
        if self._scaler_logs[last_exact.order] != last_exact:
            print(self._scaler_logs[last_exact.order])
            print(last_exact.order)
            raise ValueError("The record to be deleted is out of order. Check `_scaler_logs`")

        # Update other records to their new positions
        for x in self._scaler_logs[last_exact.order + 1 :]:
            x.order -= 1

        # Delete record
        last_exact = self._scaler_logs.pop(last_exact.order)

    def undo_all_transforms(
        self,
        domain: str | None = None,
        keys: str | list[str] | None = None,
    ):
        """
        Undo every transform applied to the specified domain/keys, in reverse order.

        Description:
            Repeatedly calls `undo_last_transform()` until:
                - all matching transforms are undone, or
                - a dependency violation prevents further undoing.

        Note:
            Raw (untransformed) data is always available via \
            `get_features(..., variant="raw")`, even without undoing.

        Args:
            domain (str | None):
                One of {"features", "targets", "tags"}. If None, reverts transforms \
                on all domains.
            keys (str | list[str] | None):
                Keys to fully revert. If None, all domain keys are used.

        Raises:
            ValueError:
                - If a transform cannot be undone because a newer transform depends on it.
                - For any other underlying error encountered by `undo_last_transform()`.

        """
        try:
            domains = [domain] if domain is not None else [FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN]
            for d in domains:
                while True:
                    self.undo_last_transform(domain=d, keys=keys)
        except ValueError as e:
            msg = str(e)

            # Case: “There are no Scalers to undo.”
            if "no scalers to undo" in msg.lower():
                return  # All transforms successfully undone

            # Case: dependency violation (“cannot be undone… depends on it”)
            if "cannot be undone" in msg.lower():
                msg = f"Stopped early. Cannot undo all transforms for: {domain}.{keys}. Reason: {msg}"
                raise ValueError(msg) from e

            # Any other ValueError is unexpected -> re-raise
            raise

    # ==========================================
    # State/Config Management Methods
    # ==========================================
    def save(self, path: str | Path, *, overwrite_existing: bool = False):
        """
        Saves FeatureSet into a single compressed Joblib file.

        Args:
            path: Filepath to save. ".mmfs.joblib" extension is added if missing.
            overwrite_existing: If False (default), raises an error if the file exists.

        """
        path = Path(path)
        if path.suffix not in (".joblib", ".mmfs", ".mmfs.joblib"):
            path = path.with_suffix(".mmfs.joblib")

        if path.exists() and not overwrite_existing:
            msg = f"{path} already exists. Use overwrite_existing=True to replace it."
            raise FileExistsError(msg)

        # Build serializable object
        obj = {
            "label": self.label,
            "table": self.collection.table,  # PyArrow table (picklable)
            "schema": self.collection.schema,  # SampleSchema (picklable)
            "scaler_logs": [r.get_state() for r in self._scaler_logs],
            "split_configs": [r.get_state() for r in self._split_configs],
        }

        joblib.dump(obj, path, compress=("lzma", 3))

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        """
        Load a FeatureSet saved via `save()`.

        Args:
            path: Path to the ".mmfs.joblib" file.

        Returns:
            FeatureSet: Fully reconstructed instance.

        """
        path = Path(path)
        if path.suffix not in (".joblib", ".mmfs", ".mmfs.joblib"):
            path = path.with_suffix(".mmfs.joblib")

        if not path.exists():
            msg = f"No FeatureSet found at: {path}"
            raise FileNotFoundError(msg)

        obj = joblib.load(path)

        # Reconstruct base FeatureSet
        fs = cls(
            label=obj["label"],
            table=obj["table"],
            schema=obj["schema"],
        )

        # Restore split configs - need to manually create splits again
        split_states = [SplitterRecord.from_state(cfg) for cfg in obj["split_configs"]]

        # Now recreate those same splits
        for split_record in split_states:
            # Get view to apply splitter to
            source = fs
            if split_record.applied_to.node != fs.label:
                raise ValueError("SplitterRecord.applied_to.node != FeatureSet.label")
            if split_record.applied_to.split is not None:
                source = fs.get_split(split_record.applied_to.split)
            # Apply splitter
            splitter = BaseSplitter.from_state(split_record.splitter_state)
            source.split(splitter=splitter, register=True, return_views=False)

        # Restore scaler logs (in sorted order)
        scaler_states = sorted(
            (ScalerRecord.from_state(cfg) for cfg in obj["scaler_logs"]),
            key=lambda rec: rec.order,
        )
        # We need to clear the "transformed" variant of the loaded collection
        # Otherwise the transform get stacked (ie applied) twice
        for domain in [FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN]:
            for k in fs.collection.get_domain_keys(domain=domain):
                if TRANSFORMED_VARIANT in fs.collection.get_variant_keys(domain=domain, key=k):
                    fs.collection.delete_variant(
                        domain=domain,
                        key=k,
                        variant=TRANSFORMED_VARIANT,
                    )
        # Now reapply the transformation (must be done after splits are created)
        for scaler_record in scaler_states:
            # Apply transform from log
            fs.fit_transform(
                scaler=scaler_record.scaler_object,
                domain=scaler_record.domain,
                keys=scaler_record.keys,
                fit_to_split=scaler_record.fit_split,
                merged_axes=scaler_record.merged_axes,
            )
        return fs
