


from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING, List

from modularml.core.data_structures.sample import Sample

class BaseSplitter(ABC):    
    @abstractmethod
    def split(self, samples:List[Sample]) -> Dict[str, List[str]]:
        """Returns a dictionary mapping subset names to `Sample.uuid`."""
        raise NotImplementedError