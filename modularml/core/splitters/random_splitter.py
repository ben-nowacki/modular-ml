

from logging import warning
from typing import TYPE_CHECKING, Dict, List
import warnings
import numpy as np

from modularml.core.data_structures.sample import Sample
from modularml.core.splitters.splitter import BaseSplitter


class RandomSplitter(BaseSplitter):
    def __init__(self, ratios: Dict[str, float], seed: int = 42):
        """
        Creates a random splitter based on sample ratios
        
        Arguments:
            ratios (Dict[str, float]): Keyword-arguments that define subset names \
                and percent splits. E.g., `RandomSplitter(train=0.5, test=0.5)`. \
                All values must add to exactly 1.0.
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
        self.seed = int(seed)
        
    def split(self, samples:List[Sample]) -> Dict[str, List[str]]:
        """
        Randomly splits a list of samples based on the defined ratios.

        Args:
            samples (List[Sample]): The list of samples to split.

        Returns:
            Dict[str, List[str]]: Dictionary mapping subset names to `Sample.uuid`.
        """
        rng = np.random.default_rng(self.seed)
        n_samples = len(samples)
        
        sample_uuids = np.asarray([
            s.uuid for s in samples
        ])
        rng.shuffle(sample_uuids)

        split_sizes = {
            key: int(round(ratio * n_samples))
            for key, ratio in self.ratios.items()
        }

        # Adjust to make sure sum of sizes equals total_samples
        size_diff = n_samples - sum(split_sizes.values())
        if size_diff != 0:
            # Adjust the largest split to make total count match
            max_key = max(split_sizes, key=split_sizes.get)
            split_sizes[max_key] += size_diff

        # Slice shuffled sample_uuids according to sizes
        splits = {}
        start = 0
        for key, size in split_sizes.items():
            splits[key] = sample_uuids[start:start + size].tolist()
            start += size
        
        return splits
    
