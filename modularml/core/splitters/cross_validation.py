from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.splitters.splitter import BaseSplitter
from modularml.utils.data_format import DataFormat

if TYPE_CHECKING:
    from modularml.core.data_structures.sample import Sample


class CrossValidationSplitter(BaseSplitter):
    """
    Implements a standard K-fold cross-validation data splitter.

    This splitter divides the dataset into *K* folds, creating paired subsets
    for training and validation for each fold. Optionally supports grouping by
    tag values to ensure that grouped samples (e.g., from the same cell or test)
    are kept within the same fold.

    Attributes:
        n_folds (int):
            Number of folds (K) to create. Must be ≥ 2.
        group_by (str | list[str] | None):
            Tag key(s) to group samples by before splitting.
            If `None`, individual samples are used for splitting.
        seed (int):
            Random seed for reproducibility of fold assignments.
        train_split_label (str):
            Label prefix for the training splits (default: `"train"`).
        val_split_label (str):
            Label prefix for the validation splits (default: `"val"`).

    Example:
        ```python
        splitter = CrossValidationSplitter(n_folds=5, group_by="cell_id", seed=42)
        folds = splitter.split(samples)
        # Example output:
        # {
        #   "train.fold_0": [...],
        #   "val.fold_0": [...],
        #   ...
        #   "train.fold_4": [...],
        #   "val.fold_4": [...]
        # }
        ```

        This creates 5 folds, each containing a unique combination of train (80%)
        and validation (20%) subsets, grouped by the `cell_id` tag.

    """

    def __init__(
        self,
        n_folds: int = 5,
        group_by: str | list[str] | None = None,
        seed: int = 13,
        train_split_label: str = "train",
        val_split_label: str = "val",
    ):
        """
        Initialize a K-fold cross-validation splitter.

        Args:
            n_folds (int, optional):
                Number of folds to create (default: 5). Must be ≥ 2.
            group_by (str | list[str] | None, optional):
                Tag key(s) to group samples by before splitting. Ensures all
                samples within a group appear in the same fold.
            seed (int, optional):
                Random seed for reproducibility (default: 13).
            train_split_label (str, optional):
                Label prefix for training folds (default: `"train"`).
            val_split_label (str, optional):
                Label prefix for validation folds (default: `"val"`).

        Raises:
            ValueError: If `n_folds < 2`.

        """
        super().__init__()

        self.n_folds = int(n_folds)
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2 for cross-validation.")

        self.group_by = group_by if isinstance(group_by, list) else [group_by] if isinstance(group_by, str) else None
        self.seed = int(seed)
        self.train_lbl = str(train_split_label)
        self.val_lbl = str(val_split_label)

    # -------------------------------------------------------------------------
    # Core logic
    # -------------------------------------------------------------------------
    def split(self, samples: list[Sample]) -> dict[str, list[str]]:
        """
        Split a list of samples into *K* folds for cross-validation.

        Args:
            samples (list[Sample]):
                List of `Sample` objects to be split.

        Returns:
            dict[str, list[str]]:
                Mapping from split names (e.g., `"train.fold_0"`, `"val.fold_0"`)
                to lists of sample UUIDs.

        Raises:
            ValueError:
                If an invalid fold count or grouping configuration is provided.

        Notes:
            - When `group_by` is `None`, splits are created on individual samples.
            - When `group_by` is specified, splits are created based on unique
              combinations of the group tags to avoid data leakage.

        """
        rng = np.random.default_rng(self.seed)
        sample_coll = SampleCollection(samples)

        # ==================================================
        # Case 1: No grouping (simple k-fold)
        # ==================================================
        if self.group_by is None:
            uuids = [s.uuid for s in samples]
            rng.shuffle(uuids)

            n_total = len(uuids)
            fold_size = n_total // self.n_folds

            folds = {}
            for f in range(self.n_folds):
                start = f * fold_size
                end = (f + 1) * fold_size if f < self.n_folds - 1 else n_total
                val_uuids = uuids[start:end]
                train_uuids = uuids[:start] + uuids[end:]
                folds[f"{self.train_lbl}.fold_{f:d}"] = train_uuids
                folds[f"{self.val_lbl}.fold_{f:d}"] = val_uuids
            return folds

        # ==================================================
        # Case 2: Grouped k-fold (e.g. by cell_id)
        # ==================================================
        df_tags = sample_coll.get_all_tags(fmt=DataFormat.PANDAS)
        df_tags["uuid"] = [s.uuid for s in samples]
        grouped = df_tags.groupby(self.group_by)

        group_keys = list(grouped.groups.keys())
        rng.shuffle(group_keys)

        n_total_groups = len(group_keys)
        fold_size = n_total_groups // self.n_folds

        folds = defaultdict(list)
        for f in range(self.n_folds):
            start = f * fold_size
            end = (f + 1) * fold_size if f < self.n_folds - 1 else n_total_groups

            val_groups = group_keys[start:end]
            train_groups = group_keys[:start] + group_keys[end:]

            # Map groups to split names
            for gk in val_groups:
                folds[f"{self.val_lbl}.fold_{f:d}"].extend(grouped.get_group((gk,))["uuid"].tolist())
            for gk in train_groups:
                folds[f"{self.train_lbl}.fold_{f:d}"].extend(grouped.get_group((gk,))["uuid"].tolist())

        return dict(folds)

    # ==================================================
    # Config serialization
    # ==================================================
    def get_config(self) -> dict[str, Any]:
        """
        Serialize the current splitter configuration.

        Returns:
            dict[str, Any]:
                Dictionary containing initialization parameters and `_target_`
                for class reconstruction.

        """
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "n_folds": self.n_folds,
            "group_by": self.group_by,
            "seed": self.seed,
            "train_split_label": self.train_lbl,
            "val_split_label": self.val_lbl,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CrossValidationSplitter:
        """
        Instantiate a CrossValidationSplitter from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary produced by `get_config()`.

        Returns:
            CrossValidationSplitter: A new instance with matching parameters.

        Raises:
            ValueError: If `_target_` does not match this class.

        """
        if "_target_" in config and config["_target_"] != f"{cls.__module__}.{cls.__name__}":
            msg = (
                f"Config _target_ does not match this class: "
                f"{config['_target_']} != {f'{cls.__module__}.{cls.__name__}'}"
            )
            raise ValueError(msg)
        return cls(
            n_folds=config["n_folds"],
            group_by=config.get("group_by"),
            seed=config.get("seed", 13),
            train_split_label=config.get("train_split_label", "train"),
            val_split_label=config.get("val_split_label", "val"),
        )
