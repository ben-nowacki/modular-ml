
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

from modularml.core.data_structures.data import Data
from modularml.utils.backend import Backend


@dataclass
class Sample:
    """
    Container for a single sample.

    Attributes:
        features (Dict[str, Any]): A set of input features. Example: {'voltage': np.ndarray}
        targets (Dict[str, Any]): A set of target values. Example: {'soh': float}
        tags (Dict[str, Any]): Metadata used for filtering, grouping, or tracking.
        
        label (str, optional): Optional user-assigned label.
        uuid (str): A globally unique ID for this sample. Automatically assigned if not provided.
    """
    features: Dict[str, Data]
    targets: Dict[str, Data]
    tags: Dict[str, Data]
    
    label: Optional[str] = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if not isinstance(self.uuid, str):
            raise TypeError(f"Sample ID must be a string. Got: {type(self.uuid)}")
        
        # Check Data type
        for k, v in self.features.items():
            if not isinstance(v, Data):
                raise TypeError(f"Feature '{k}' is not a Data object: {type(v)}")
        for k, v in self.targets.items():
            if not isinstance(v, Data):
                raise TypeError(f"Target '{k}' is not a Data object: {type(v)}")
        for k, v in self.tags.items():
            if not isinstance(v, Data):
                raise TypeError(f"Tag '{k}' is not a Data object: {type(v)}")
    
        # Check for backend consistency
        data_items = list(self.features.values()) + list(self.targets.values())
        backends = {d.backend for d in data_items if isinstance(d, Data)}

        if len(backends) > 1:
            # Choose a target backend (e.g., SCITKIT by default)
            target_backend = Backend.SCIKIT
            self.features = {k: v.as_backend(target_backend) for k, v in self.features.items()}
            self.targets = {k: v.as_backend(target_backend) for k, v in self.targets.items()}
            
        # Enforce consistent shapes
        f_shapes = [d.shape for d in self.features.values()]
        if len(set(f_shapes)) > 1:
            raise ValueError(f"Inconsistent feature shapes: {f_shapes}")
        self._feature_shape = (len(f_shapes), ) + tuple(f_shapes[0])
        
        t_shapes = [d.shape for d in self.targets.values()]
        if len(set(t_shapes)) > 1:
            raise ValueError(f"Inconsistent target shapes: {t_shapes}")
        self._target_shape = (len(t_shapes), ) + tuple(t_shapes[0])
        
        
    def __repr__(self) -> str:
        def summarize(d: Dict[str, Any]) -> Dict[str, str]:
            return {k: type(v).__name__ for k, v in d.items()}
        
        return (
            f"Sample("
            f"features={summarize(self.features)}, "
            f"targets={summarize(self.targets)}, "
            f"tags={summarize(self.tags)}, "
            f"label={self.label}, "
            f"id={self.uuid[:8]}..."
            f")"
        )
        
    @property
    def feature_keys(self) -> List[str]:
        return list(self.features.keys())
    
    @property
    def target_keys(self) -> List[str]:
        return list(self.targets.keys())
    
    @property
    def tag_keys(self) -> List[str]:
        return list(self.tags.keys())
    
    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return self._feature_shape

    @property
    def target_shape(self) -> Tuple[int, ...]:
        return self._target_shape

    def get_features(self, key: str) -> Data:
        return self.features.get(key)

    def get_targets(self, key: str) -> Data:
        return self.targets.get(key)

    def get_tags(self, key: str) -> Data:
        return self.tags.get(key)


    def to_backend(self, backend: Backend) -> "Sample":
        return Sample(
            features={k: v.to_backend(backend) for k, v in self.features.items()},
            targets={k: v.to_backend(backend) for k, v in self.targets.items()},
            tags={k: v.to_backend(backend) for k, v in self.tags.items()},
            label=self.label,
            uuid=self.uuid
        )


