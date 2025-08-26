

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd

from modularml.core.data_structures.sample import Sample, SampleCollection
from modularml.utils.data_format import DATA_FORMATS, convert_to_format


class SampleAttribute(str, Enum):
    FEATURES = "features"
    TARGETS = "targets"
    TAGS = "tags"

@dataclass(frozen=True) 
class BatchComponentSelector:
    """
    A selector class to wrap accessing specific components
    of the Batch object.
    
    Attributes:
        role (str): The role within Batch to use
        sample_attribute (SampleAttribute): The attribute of Sample to use.
        attribute_key (str, optional): An optional subset of the specified Sample.sample_attribute. 
            E.g., if Sample.features contains {'voltage':..., 'current':...}, you can access just \
            the 'voltage' component using: `sample_attribute='features', attribute_key='voltage'`
    """
    role: str
    sample_attribute: SampleAttribute
    attribute_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.sample_attribute in SampleAttribute:
            raise ValueError(
                f"`sample_attribute` must be one of the following: {SampleAttribute._member_names_}"
                f"Received: {self.sample_attribute}."
            )
    
    def to_string(self) -> str:
        return f"{self.role}.{self.sample_attribute.value}.{self.attribute_key}"


    def get_config(self) -> Dict[str, Any]:
        cfg = {
            "role": str(self.role),
            "sample_attribute": str(self.sample_attribute.value),
        }
        if self.attribute_key is not None:
            cfg['attribute_key'] = str(self.attribute_key)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchComponentSelector":
        return cls(
            role=config['role'],
            sample_attribute=SampleAttribute(config['sample_attribute']),
            attribute_key=config.get('attribute_key', None)
        )
    
    
    
        
@dataclass
class Batch(SampleCollection): 
    """
    Container for a single batch of samples

    Attributes:
        _samples (Dict[str, List[Sample]]): List of samples contained in this \
            batch assigned to a string-based "role". E.g., for triplet-based \
            batches, you'd have \
            `_samples={'anchor':List[Sample], 'negative':List[Sample], ...}`.
        _sample_weights: (Dict[str, List[float]]): List of weights  applied to \
            samples in this batch, using the same string-based "role" dictionary. \
            E.g., `_sample_weights={'anchor':List[float], 'negative':..., ...}`. \
            If None, all samples will have the same weight.
        index (int, optional): Optional user-assigned index.
        batch_id (str): A globally unique ID for this batch. Automatically assigned if not provided.    
    """  
    _samples: Dict[str, List[Sample]]
    _sample_weights: Dict[str, List[float]] = None
    index: Optional[int] = None
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if not isinstance(self.batch_id, str):
            raise TypeError(f"Batch ID must be a string. Got: {type(self.batch_id)}")
        
        
        # check sample shapes (features & targets) for each role
        role_shapes = {'features':{}, 'targets':{}}
        for role, role_samples in self._samples.items():
            # enforce feature shape within role
            f_shapes = list(set([s.feature_shape for s in role_samples]))
            if not len(f_shapes) == 1:
                raise ValueError(f"Batch samples have inconsistent feature sizes: {f_shapes}.")
            
            # enforce target shape within role
            t_shapes = list(set([s.target_shape for s in role_samples]))
            if not len(t_shapes) == 1:
                raise ValueError(f"Batch samples have inconsistent targer sizes: {t_shapes}.")
            
            role_shapes['features'][role] = f_shapes[0]
            role_shapes['targets'][role] = t_shapes[0]
            
        # enforce each shapes across roles
        for k in ['features', 'targets']:
            k_shapes = set(role_shapes[k].values())
            if not len(k_shapes) == 1:
                raise ValueError(f"Batch samples have inconsistent role {k} sizes: {k_shapes}.")
            
        # check weight shapes
        if self._sample_weights is None:
            self._sample_weights = {
                r: [1, ] * self.n_samples
                for r in self._samples.keys()
            }
        else:
            # check that each sample weight key matches sample length 
            for role in self._samples.keys():
                if not role in self._sample_weights.keys():
                    raise KeyError(f'Batch sample weights is missing required role: `{role}`')
                
                if not len(self._sample_weights[role]) == self.n_samples:
                    raise ValueError(
                        f"Length of batch sample weights does not match length of samples "
                        f"for role `{role}`: {len(self._sample_weights[role])} != {self.n_samples}."
                    )
        
        
    @property
    def samples(self):
        """A flattened view of all samples contained in this batch."""
        flat_samples : List[Sample] = []
        for role, role_samples in self._samples.items():
            flat_samples.extend(role_samples)
        return flat_samples
    
    @property
    def formatted_samples(self) -> Dict[str, List[Sample]]:
        """
        A structured view of all samples in this batch. 
        Returns each role and the corresponding sample list.
        """
        return self._samples 
    
    @property
    def available_roles(self) -> List[str]:
        """All assigned roles in this batch."""
        return list(self._samples.keys())

    @property
    def n_samples(self) -> int:
        return len(self._samples[self.available_roles[0]])
    
    @property
    def n_roles(self) -> int:
        return len(self._samples)
    
    
    def __len__(self):
        return self.n_samples

    def unpack(self, format: DATA_FORMATS = 'dict') -> Dict[str, Tuple[Any, Any, Any]]:
        """
        Unpacks the Batch of Sample objects into separate feature, target, and tags.
        Each element of tuple will be return in the specified data `format`.

        Args:
            format (DATA_FORMATS, optional): The format of each unpacked sample \
                attribute. E.g., `format='df'` will return the features, targets, \
                and tags as individual dataframes. Defaults to 'dict'.
                
        Returns:
            Dict[str, Tuple[Any, Any, Any]]: Unpacked attributes assigned to each \
                role in this batch.
        """
        
        unpacked : Dict[str, Tuple[Any, Any, Any]] = {}
        
        for role, samples in self.formatted_samples.items():
            temp = SampleCollection(samples=samples)
            unpacked[role] = (
                temp.get_all_features(format=format),
                temp.get_all_targets(format=format),
                temp.get_all_tags(format=format),
            )
            
        return unpacked
    
    def select(self, selection:BatchComponentSelector, format: DATA_FORMATS = 'list') -> Any:
        """Select the specified components from this Batch."""
    
        if not selection.role in self.available_roles:
            raise KeyError(
                f"`selection.role='{selection.role}'` does not exist in this batch. "
                f"Available roles include: {self.available_roles}"
            )
        
        selected_samples = self.formatted_samples[selection.role]
        temp = SampleCollection(samples=selected_samples)
        
        if selection.sample_attribute == SampleAttribute.FEATURES:
            if selection.attribute_key is None:
                return temp.get_all_features(format=format)
            else:
                all_features : pd.DataFrame = temp.get_all_features(format='df')
                if not selection.attribute_key in all_features.columns:
                    raise KeyError(
                        f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
                        f"the Sample.{selection.sample_attribute} values."
                        f"Available attribute_keys include: {list(all_features.columns)}"
                    )
                    
                return convert_to_format(
                    data={selection.to_string(): all_features[selection.attribute_key].values},
                    format=format
                )
        
        elif selection.sample_attribute == SampleAttribute.TARGETS:
            if selection.attribute_key is None:
                return temp.get_all_targets(format=format)
            else:
                all_targets : pd.DataFrame = temp.get_all_targets(format='df')
                if not selection.attribute_key in all_targets.columns:
                    raise KeyError(
                        f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
                        f"the Sample.{selection.sample_attribute} values."
                        f"Available attribute_keys include: {list(all_targets.columns)}"
                    )
                
                return convert_to_format(
                    data={selection.to_string(): all_targets[selection.attribute_key].values},
                    format=format
                )
        
        elif selection.sample_attribute == SampleAttribute.TAGS:
            if selection.attribute_key is None:
                return temp.get_all_tags(format=format)
            else:
                all_tags : pd.DataFrame = temp.get_all_tags(format='df')
                if not selection.attribute_key in all_tags.columns:
                    raise KeyError(
                        f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
                        f"the Sample.{selection.sample_attribute} values."
                        f"Available attribute_keys include: {list(all_tags.columns)}"
                    )
                
                return convert_to_format(
                    data={selection.to_string(): all_tags[selection.attribute_key].values},
                    format=format
                )

        else:
            raise ValueError(f"Unknown value of `selection.sample_attribute`: {selection.sample_attribute}")





# @dataclass
# class Batch(SampleCollection):
#     """
#     Container for a single batch

#     Attributes:
#         _all_samples (Union[List[Sample], Tuple[List[Sample]], Dict[List[Sample]]]):

#         batch_id (str): A globally unique ID for this batch. Automatically assigned if not provided.
#         samples (List[Sample]): The samples contained in this batch
#         index (int, optional): Optional user-assigned index.
#     """    
#     samples: List[Sample]
#     index: Optional[int] = None
#     batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

#     def __post_init__(self):
#         if not isinstance(self.batch_id, str):
#             raise TypeError(f"Batch ID must be a string. Got: {type(self.batch_id)}")

#     def __repr__(self) -> str:
#         return (
#             f"Batch("
#             f"id={self.batch_id[:8]}..., "
#             f"n_samples={len(self.samples)}, "
#             f"index={self.index}"
#             f")"
#         )

#     def __len__(self):
#         return len(self.samples)

#     def unpack(self, format: DATA_FORMATS = 'dict') -> Tuple[Any, Any, Any]:
#         """
#         Unpacks the Batch of Sample objects into separate feature, target, and tag dicts.
#         Each element of tuple will be return in the specified data `format`.

#         Returns:
#             Tuple[features, targets, tags]
#         """
        
#         return (
#             self.get_all_features(format=format),
#             self.get_all_targets(format=format),
#             self.get_all_tags(format=format),
#         )
