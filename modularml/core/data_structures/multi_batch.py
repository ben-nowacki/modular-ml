
from dataclasses import dataclass
from typing import Dict
from modularml.core.data_structures.batch import Batch


@dataclass
class MultiBatch:
    """Container for batches from multiple FeatureSets, keyed by FeatureSet.label"""
    batches: Dict[str, Batch]


    def __getitem__(self, key: str) -> Batch:
        return self.batches[key]
    
    def get(self, label: str) -> Batch:
        return self.batches[label]

    def items(self):
        return self.batches.items()
    
    def keys(self):
        return self.batches.keys()
    
    def __contains__(self, key: str) -> bool:
        return key in self.batches
