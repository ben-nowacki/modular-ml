

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.utils.data_format import DataFormat, convert_to_format


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
    

    






    # def unpack(self, format: DATA_FORMATS = 'dict') -> Dict[str, Tuple[Any, Any, Any]]:
    #     """
    #     Unpacks the Batch of Sample objects into separate feature, target, and tags.
    #     Each element of tuple will be return in the specified data `format`.

    #     Args:
    #         format (DATA_FORMATS, optional): The format of each unpacked sample \
    #             attribute. E.g., `format='df'` will return the features, targets, \
    #             and tags as individual dataframes. Defaults to 'dict'.
                
    #     Returns:
    #         Dict[str, Tuple[Any, Any, Any]]: Unpacked attributes assigned to each \
    #             role in this batch.
    #     """
        
    #     unpacked : Dict[str, Tuple[Any, Any, Any]] = {}
        
    #     for role, samples in self.formatted_samples.items():
    #         temp = SampleCollection(samples=samples)
    #         unpacked[role] = (
    #             temp.get_all_features(format=format),
    #             temp.get_all_targets(format=format),
    #             temp.get_all_tags(format=format),
    #         )
            
    #     return unpacked
    
    # def select(self, selection:BatchComponentSelector, format: DATA_FORMATS = 'list') -> Any:
    #     """Select the specified components from this Batch."""
    
    #     if not selection.role in self.available_roles:
    #         raise KeyError(
    #             f"`selection.role='{selection.role}'` does not exist in this batch. "
    #             f"Available roles include: {self.available_roles}"
    #         )
        
    #     selected_samples = self.formatted_samples[selection.role]
    #     temp = SampleCollection(samples=selected_samples)
        
    #     if selection.sample_attribute == SampleAttribute.FEATURES:
    #         if selection.attribute_key is None:
    #             return temp.get_all_features(format=format)
    #         else:
    #             all_features : pd.DataFrame = temp.get_all_features(format='df')
    #             if not selection.attribute_key in all_features.columns:
    #                 raise KeyError(
    #                     f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
    #                     f"the Sample.{selection.sample_attribute} values."
    #                     f"Available attribute_keys include: {list(all_features.columns)}"
    #                 )
                    
    #             return convert_to_format(
    #                 data={selection.to_string(): all_features[selection.attribute_key].values},
    #                 format=format
    #             )
        
    #     elif selection.sample_attribute == SampleAttribute.TARGETS:
    #         if selection.attribute_key is None:
    #             return temp.get_all_targets(format=format)
    #         else:
    #             all_targets : pd.DataFrame = temp.get_all_targets(format='df')
    #             if not selection.attribute_key in all_targets.columns:
    #                 raise KeyError(
    #                     f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
    #                     f"the Sample.{selection.sample_attribute} values."
    #                     f"Available attribute_keys include: {list(all_targets.columns)}"
    #                 )
                
    #             return convert_to_format(
    #                 data={selection.to_string(): all_targets[selection.attribute_key].values},
    #                 format=format
    #             )
        
    #     elif selection.sample_attribute == SampleAttribute.TAGS:
    #         if selection.attribute_key is None:
    #             return temp.get_all_tags(format=format)
    #         else:
    #             all_tags : pd.DataFrame = temp.get_all_tags(format='df')
    #             if not selection.attribute_key in all_tags.columns:
    #                 raise KeyError(
    #                     f"`selection.attribute_key='{selection.attribute_key}'` does not exist in "
    #                     f"the Sample.{selection.sample_attribute} values."
    #                     f"Available attribute_keys include: {list(all_tags.columns)}"
    #                 )
                
    #             return convert_to_format(
    #                 data={selection.to_string(): all_tags[selection.attribute_key].values},
    #                 format=format
    #             )

    #     else:
    #         raise ValueError(f"Unknown value of `selection.sample_attribute`: {selection.sample_attribute}")

