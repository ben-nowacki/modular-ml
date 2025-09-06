from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.splitters.splitter import BaseSplitter
from modularml.utils.data_format import DataFormat

if TYPE_CHECKING:
    from modularml.core.data_structures.sample import Sample


class RandomSplitter(BaseSplitter):
    def __init__(self, ratios: dict[str, float], group_by: str | list[str] | None = None, seed: int = 42):
        """
        Creates a random splitter based on sample ratios.

        Arguments:
            ratios (dict[str, float]): Keyword-arguments that define subset names \
                and percent splits. E.g., `RandomSplitter(train=0.5, test=0.5)`. \
                All values must add to exactly 1.0.
            group_by (Union[str, list[str]], optional): Tag key(s) to group samples \
                by before splitting.
            seed (int): The seed of the random generator.

        """
        super().__init__()

        total = 0.0
        for v in ratios.values():
            total += float(v)

        if total != 1.0:
            msg = f"ratios must sum to exactly 1.0. Total = {total}"
            raise ValueError(msg)

        self.ratios = ratios
        self.group_by = (
            group_by
            if isinstance(group_by, list)
            else [
                group_by,
            ]
            if isinstance(group_by, str)
            else None
        )

        self.seed = int(seed)

    def _get_split_boundaries(self, n: int) -> dict[str, tuple]:
        """Returns index boundaries for each subset split (for n elements)."""
        boundaries = {}
        current = 0
        for i, (split_name, ratio) in enumerate(self.ratios.items()):
            count = int(ratio * n)
            if i == len(self.ratios) - 1:
                count = n - current
            boundaries[split_name] = (current, current + count)
            current += count
        return boundaries

    def split(self, samples: list[Sample]) -> dict[str, list[str]]:
        """
        Splits samples into subsets according to ratios and grouping.

        Args:
            samples (list[Sample]): The list of samples to split.

        Returns:
            dict[str, list[str]]: Mapping from subset name to list of Sample.uuid.

        """
        sample_coll = SampleCollection(samples)
        rng = np.random.default_rng(self.seed)

        if self.group_by is None:
            uuids = [s.uuid for s in samples]
            rng.shuffle(uuids)

            n_total = len(uuids)
            boundaries = self._get_split_boundaries(n_total)

            split_result = {}
            for split_name, (start, end) in boundaries.items():
                split_result[split_name] = uuids[start:end]

            return split_result

        # Group by tags
        df_tags = sample_coll.get_all_tags(format=DataFormat.PANDAS)
        df_tags["uuid"] = [s.uuid for s in samples]
        grouped = df_tags.groupby(self.group_by)

        group_keys = list(grouped.groups.keys())
        rng.shuffle(group_keys)

        n_total = len(group_keys)
        boundaries = self._get_split_boundaries(n_total)

        group_to_split = {}
        for split_name, (start, end) in boundaries.items():
            for gk in group_keys[start:end]:
                group_to_split[gk] = split_name

        split_result = defaultdict(list)
        for _, row in df_tags.iterrows():
            group_key = tuple(row[k] for k in self.group_by) if len(self.group_by) > 1 else row[self.group_by[0]]
            split_name = str(group_to_split[group_key])
            split_result[split_name].append(row["uuid"])

        return dict(split_result)

    def get_config(
        self,
    ) -> dict[str, Any]:
        """Returns a configuration to reproduce identical split configurations."""
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "ratios": self.ratios,
            "group_by": self.group_by,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RandomSplitter:
        """Instantiates a RandomSplitter from config."""
        if "_target_" in config and config["_target_"] != f"{cls.__module__}.{cls.__name__}":
            msg = (
                f"Config _target_ does not match this class: "
                f"{config['_target_']} != {f'{cls.__module__}.{cls.__name__}'}"
            )
            raise ValueError(msg)

        return cls(ratios=config["ratios"], group_by=config["group_by"], seed=config["seed"])
