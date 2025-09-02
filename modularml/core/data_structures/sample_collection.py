



from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_dict_to_format, convert_to_format, enforce_numpy_shape, format_has_shape


@dataclass
class SampleCollection:
    """
    A lightweight container for a list of Sample instances.

    Attributes:
        samples (List[Sample]): A list of Sample instances.
    """
    
    samples: List[Sample]
    
    def __post_init__(self):
        if not all(isinstance(s, Sample) for s in self.samples):
            raise TypeError("All elements in `samples` must be of type Sample.")
        if len(self.samples) == 0:
            raise ValueError("SampleCollection must contain at least one Sample.")

        self._uuid_sample_map: Dict[str, Sample] = {
            s.uuid: s for s in self.samples
        }
        self._label_sample_map: Dict[str, Sample] = {
            s.label: s for s in self.samples
        }
        
        # Enforce consistent shapes
        f_shapes = list(set([s.feature_shape for s in self.samples]))
        if not len(f_shapes) == 1:
            raise ValueError(f"Inconsistent SampleCollection feature shapes: {f_shapes}.")
            
        t_shapes = list(set([s.target_shape for s in self.samples]))
        if not len(t_shapes) == 1:
            raise ValueError(f"Inconsistent SampleCollection target sizes: {t_shapes}.")
        
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]
    
    def get_sample_with_uuid(self, uuid:str) -> Sample:
        return self._uuid_sample_map[uuid]
    def get_samples_with_uuid(self, uuids: List[str]) -> "SampleCollection":
        return SampleCollection(samples=[
            self.get_sample_with_uuid(x)
            for x in uuids
        ])
    
    def get_sample_with_label(self, label:str) -> Sample:
        return self._label_sample_map[label]
    def get_samples_with_label(self, labels: List[str]) -> "SampleCollection":
        return SampleCollection(samples=[
            self.get_sample_with_uuid(x)
            for x in labels
        ])
    
    
    def __iter__(self):
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"SampleCollection(n_samples={len(self.samples)})"

    @property
    def feature_keys(self) -> List[str]:
        return self.samples[0].feature_keys
    
    @property
    def target_keys(self) -> List[str]:
        return self.samples[0].target_keys
    
    @property
    def tag_keys(self) -> List[str]:
        return self.samples[0].tag_keys
    
    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return self.samples[0].feature_shape

    @property
    def target_shape(self) -> Tuple[int, ...]:
        return self.samples[0].target_shape

    @property
    def sample_uuids(self) -> List[str]:
        return list(self._uuid_sample_map.keys())
    
    @property
    def sample_labels(self) -> List[str]:
        return list(self._label_sample_map.keys())

    
    def get_all_features(self, format: Union[str, DataFormat] = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all features across all samples in the specified format.
        """ 
        raw = {
            k: [s.features[k].value for s in self.samples]
            for k in self.feature_keys
        }
        if format_has_shape(format=format):
            # force proper shape: conver to np, force shape, then convert to specific format
            np_data = convert_dict_to_format(raw, format=DataFormat.NUMPY)
            np_data = enforce_numpy_shape(
                arr=np_data, 
                target_shape=(len(self), *self.feature_shape)
            )
            return convert_to_format(np_data, format=format)
        
        return convert_dict_to_format(raw, format=format)
    
    def get_all_targets(self, format: Union[str, DataFormat] = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all targets across all samples in the specified format.
        """
        raw = {
            k: [s.targets[k].value for s in self.samples]
            for k in self.target_keys
        }
        if format_has_shape(format=format):
            # force proper shape: conver to np, force shape, then convert to specific format
            np_data = convert_dict_to_format(raw, format=DataFormat.NUMPY)
            np_data = enforce_numpy_shape(
                arr=np_data, 
                target_shape=(len(self), *self.target_shape)
            )
            return convert_to_format(np_data, format=format)

        return convert_dict_to_format(raw, format=format)

    def get_all_tags(self, format: Union[str, DataFormat] = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all tags across all samples in the specified format.

        Each tag will be returned as a list of values across samples.
        The format argument controls the output structure.
        """
        raw = {
            k: [s.tags[k].value for s in self.samples]
            for k in self.tag_keys
        }
        return convert_dict_to_format(raw, format=format)


    def to_backend(self, backend: Union[str, Backend]) -> "SampleCollection":
        """
        Returns a new SampleCollection with all Data objects converted to the specified backend.
        """
        if isinstance(backend, str):
            backend = Backend(backend)
            
        converted = []
        for s in self.samples:
            features = {k: v.to_backend(backend) for k, v in s.features.items()}
            targets = {k: v.to_backend(backend) for k, v in s.targets.items()}
            tags = {k: v.to_backend(backend) for k, v in s.tags.items()}
            converted.append(Sample(features=features, targets=targets, tags=tags, index=s.index, sample_id=s.sample_id))
        return SampleCollection(samples=converted)



