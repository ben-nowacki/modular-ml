

from collections import defaultdict
from logging import warning
from typing import TYPE_CHECKING, Dict, List, Union
import warnings
import numpy as np

from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.splitters.splitter import BaseSplitter
from modularml.utils.data_format import DataFormat


class RandomSplitter(BaseSplitter):
    def __init__(self, ratios: Dict[str, float], group_by: Union[str, List[str]] = None, seed: int = 42):
        """
        Creates a random splitter based on sample ratios
        
        Arguments:
            ratios (Dict[str, float]): Keyword-arguments that define subset names \
                and percent splits. E.g., `RandomSplitter(train=0.5, test=0.5)`. \
                All values must add to exactly 1.0.
            group_by (Union[str, List[str]], optional): Tag key(s) to group samples \
                by before splitting.
            seed (int): The seed of the random generator.
        """
        super().__init__()
        
        total = 0.0
        for k,v in ratios.items():
            try: 
                v = float(v)
            except ValueError:
                raise TypeError(f"ratio values must be a float. Recevied: type({v})={type(v)}")
            
            total += v
        if not total == 1.0:
            raise ValueError(f"ratios must sum to exactly 1.0. Total = {total}")
        
        self.ratios = ratios        
        self.group_by = group_by if isinstance(group_by, list) else [group_by, ] if isinstance(group_by, str) else None

        self.seed = int(seed)
        
    def _get_split_boundaries(self, n: int) -> Dict[str, tuple]:
        """
        Returns index boundaries for each subset split (for n elements).
        """
        boundaries = {}
        current = 0
        for i, (split_name, ratio) in enumerate(self.ratios.items()):
            count = int(ratio * n)
            if i == len(self.ratios) - 1:
                count = n - current
            boundaries[split_name] = (current, current + count)
            current += count
        return boundaries
    
    
    
    def split(self, samples:List[Sample]) -> Dict[str, List[str]]:
        """
        Splits samples into subsets according to ratios and grouping.

        Args:
            samples (List[Sample]): The list of samples to split.
            
        Returns:
            Dict[str, List[str]]: Mapping from subset name to list of Sample.uuid.
        """
        sample_coll = SampleCollection(samples)
        rng = np.random.default_rng(self.seed)
        
        if self.group_by is None:
            uuids = [s.uuid for s in samples]
            rng.shuffle(uuids)

            n_total = len(uuids)
            boundaries = self._get_split_boundaries(n_total)

            split_result = {}
            for i, (split_name, (start, end)) in enumerate(boundaries.items()):
                split_result[split_name] = uuids[start:end]

            return split_result
        
        else:
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