
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection

 
@dataclass
class Batch: 
    """
    Container for a single batch of samples

    Attributes:
        role_samples (Dict[str, SampleCollection]): Sample collections in \
            batch assigned to a string-based "role". E.g., for triplet-based \
            batches, you'd have \
            `_samples={'anchor':List[Sample], 'negative':List[Sample], ...}`.
        role_sample_weights: (Dict[str, Data]): List of weights  applied to \
            samples in this batch, using the same string-based "role" dictionary. \
            E.g., `_sample_weights={'anchor':List[float], 'negative':..., ...}`. \
            If None, all samples will have the same weight.
        label (str, optional): Optional user-assigned label.
        uuid (str): A globally unique ID for this batch. Automatically assigned if not provided.    
    """  
    role_samples: Dict[str, SampleCollection]
    role_sample_weights: Dict[str, Data] = None
    label: Optional[str] = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        # Enforce consistent shapes
        f_shapes = list(set([c.feature_shape for c in self.role_samples.values()]))
        if not len(f_shapes) == 1:
            raise ValueError(f"Inconsistent feature shapes across Batch roles: {f_shapes}.")
        self._feature_shape = f_shapes[0]
        
        t_shapes = list(set([c.target_shape for c in self.role_samples.values()]))
        if not len(t_shapes) == 1:
            raise ValueError(f"Inconsistent target shapes across Batch roles: {t_shapes}.")
        self._target_shape = t_shapes[0]
        
        # Check weight shapes
        if self.role_sample_weights is None:
            self.role_sample_weights = {
                r: Data([1, ] * len(c))
                for r,c in self.role_samples.items()
            }
        else:
            # Check that each sample weight key matches sample length 
            for r, c in self.role_samples.items():
                if not r in self.role_sample_weights.keys():
                    raise KeyError(f'Batch `role_sample_weights` is missing required role: `{r}`')
                if not len(self.role_sample_weights[r]) == len(c):
                    raise ValueError(
                        f"Length of batch sample weights does not match length of samples "
                        f"for role `{r}`: {len(self.role_sample_weights[r])} != {len(c)}."
                    )
        
    @property
    def available_roles(self) -> List[str]:
        """All assigned roles in this batch."""
        return list(self.role_samples.keys())

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return self._feature_shape
    
    @property
    def target_shape(self) -> Tuple[int, ...]:
        return self._target_shape

    @property
    def n_samples(self) -> int:
        if not hasattr(self, "_n_samples") or self._n_samples is None:
            self._n_samples = len(self.role_samples[self.available_roles[0]])
        return self._n_samples
       
    def __len__(self):
        return self.n_samples

    def get_samples(self, role:str) -> SampleCollection:
        return self.role_samples[role]
    

@dataclass
class BatchOutput:
    features: Dict[str, Any]
    sample_uuids: Dict[str, Any]
    targets: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, Any]] = None 
    
    
    def __post_init__(self, ): 
        # Enforce consistent shapes
        f_shapes = list(set([self.features[role].shape for role in self.features.keys()]))
        if not len(f_shapes) == 1:
            raise ValueError(f"Inconsistent feature shapes across BatchOutput roles: {f_shapes}.")
        self._feature_shape = f_shapes[0]
        
        if self.targets is not None:
            t_shapes = list(set([self.targets[role].shape for role in self.targets.keys()]))
            if not len(t_shapes) == 1:
                raise ValueError(f"Inconsistent target shapes across Batch roles: {t_shapes}.")
            self._target_shape = t_shapes[0]
        else:  self._target_shape = None
        
        if self.tags is not None:
            t_shapes = list(set([self.tags[role].shape for role in self.tags.keys()]))
            if not len(t_shapes) == 1:
                raise ValueError(f"Inconsistent tag shapes across Batch roles: {t_shapes}.")
            self._target_shape = t_shapes[0]
        else:  self._target_shape = None
        
        # Ensure feature keys = sample uuid keys
        f_keys = set(list(self.features.keys()))
        s_keys = set(list(self.sample_uuids.keys()))
        if f_keys.difference(s_keys):
            raise ValueError(f"features and sample_uuids have differing keys: {f_keys} != {s_keys}.")
        
    
    def to_batch(self, label: str = None, role_sample_weights = None) -> Batch:
        role_samples = {}
        
        for role in self.available_roles:
            samples = []
            for i in range(len(self.features[role])):
                features = {
                    f"output_{i}": Data(self.features[role][i][j])
                    for j in range(len(self.features[role][i]))
                }
                targets = self.targets
                if self.targets is not None:
                    targets = {
                        f"output_{i}": Data(self.targets[role][i][j])
                        for j in range(len(self.targets[role][i]))
                    }
                tags = self.tags
                if self.tags is not None:
                    tags = {
                        f"output_{i}": Data(self.tags[role][i][j])
                        for j in range(len(self.tags[role][i]))
                    }
                metadata = self.metadata
                if self.metadata is not None:
                    metadata = {
                        f"output_{i}": Data(self.metadata[role][i][j])
                        for j in range(len(self.metadata[role][i]))
                    }
                    
                samples.append(Sample(
                    features=features, targets=targets, tags=tags, metadata=metadata
                ))
            
            role_samples[role] = SampleCollection(samples)
        
        return Batch(
            role_samples=role_samples,
            role_sample_weights=role_sample_weights,
            label=label,
        )
    
    @property
    def available_roles(self) -> List[str]:
        """All assigned roles."""
        return list(self.features.keys())

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return self._feature_shape
    
    @property
    def target_shape(self) -> Tuple[int, ...]:
        return self._target_shape