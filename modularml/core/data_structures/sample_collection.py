



from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_to_format


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
    
    def get_sample_with_label(self, label:str) -> Sample:
        return self._label_sample_map[label]
    
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
        raw = {}

        for k in self.feature_keys:
            values = [s.features[k].value for s in self.samples]

            if format in ("dict_numpy", DataFormat.DICT_NUMPY):
                try:
                    raw[k] = np.array(values)
                except Exception:
                    raw[k] = values
            else:
                raw[k] = values

        return convert_to_format(raw, format=format)

    def get_all_targets(self, format: Union[str, DataFormat] = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all targets across all samples in the specified format.
        """
        target_keys = self.samples[0].targets.keys()
        raw = {}

        for k in target_keys:
            values = [s.targets[k].value for s in self.samples]

            if format in ("dict_numpy", DataFormat.DICT_NUMPY):
                try:
                    raw[k] = np.array(values)
                except Exception:
                    raw[k] = values
            else:
                raw[k] = values

        return convert_to_format(raw, format=format)

    def get_all_tags(self, format: Union[str, DataFormat] = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all tags across all samples in the specified format.

        Each tag will be returned as a list of values across samples.
        The format argument controls the output structure.
        """
        raw : Dict[str, Any] = {}

        for key in self.samples[0].tags:
            values = [
                v.value.item() if isinstance(v.value, np.ndarray) and v.value.size == 1 else v.value
                for v in [s.tags[key] for s in self.samples]
            ]

            if format in ("dict_numpy", DataFormat.DICT_NUMPY):
                try:
                    raw[key] = np.array(values)
                except Exception:
                    raw[key] = values
            else:
                raw[key] = values

        return convert_to_format(raw, format=format)


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



