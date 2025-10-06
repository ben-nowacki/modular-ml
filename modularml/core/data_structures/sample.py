from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from modularml.core.data_structures.data import Data
from modularml.utils.backend import Backend


@dataclass
class Sample:
    """
    Container for a single sample.

    Attributes:
        features (dict[str, Any]): A set of input features. Example: {'voltage': np.ndarray}
        targets (dict[str, Any]): A set of target values. Example: {'soh': float}
        tags (dict[str, Any]): Metadata used for filtering, grouping, or tracking.

        label (str, optional): Optional user-assigned label.
        uuid (str): A globally unique ID for this sample. Automatically assigned if not provided.

    """

    features: dict[str, Data]
    targets: dict[str, Data]
    tags: dict[str, Data]

    label: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not isinstance(self.uuid, str):
            msg = f"Sample ID must be a string. Got: {type(self.uuid)}"
            raise TypeError(msg)

        # Check Data type
        for k, v in self.features.items():
            if not isinstance(v, Data):
                msg = f"Feature '{k}' is not a Data object: {type(v)}"
                raise TypeError(msg)
        for k, v in self.targets.items():
            if not isinstance(v, Data):
                msg = f"Target '{k}' is not a Data object: {type(v)}"
                raise TypeError(msg)
        for k, v in self.tags.items():
            if not isinstance(v, Data):
                msg = f"Tag '{k}' is not a Data object: {type(v)}"
                raise TypeError(msg)

        # Check for backend consistency
        data_items = list(self.features.values()) + list(self.targets.values())
        backends = {d.backend for d in data_items if isinstance(d, Data)}

        if len(backends) > 1:
            # Choose a target backend (e.g., SCITKIT by default)
            target_backend = Backend.SCIKIT
            self.features = {k: v.as_backend(target_backend) for k, v in self.features.items()}
            self.targets = {k: v.as_backend(target_backend) for k, v in self.targets.items()}

        # Enforce consistent shapes?
        # f_shapes = [d.shape for d in self.features.values()]
        # if len(set(f_shapes)) > 1:
        #     msg = f"Inconsistent feature shapes: {f_shapes}"
        #     raise ValueError(msg)
        # self._feature_shape = (len(f_shapes), *tuple(f_shapes[0]))

        # t_shapes = [d.shape for d in self.targets.values()]
        # if len(set(t_shapes)) > 1:
        #     msg = f"Inconsistent target shapes: {t_shapes}"
        #     raise ValueError(msg)
        # self._target_shape = (len(t_shapes), *tuple(t_shapes[0]))

    def __repr__(self) -> str:
        def summarize(d: dict[str, Any]) -> dict[str, str]:
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
    def feature_keys(self) -> list[str]:
        return list(self.features.keys())

    @property
    def target_keys(self) -> list[str]:
        return list(self.targets.keys())

    @property
    def tag_keys(self) -> list[str]:
        return list(self.tags.keys())

    @property
    def feature_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(v.shape for v in self.features.values())

    @property
    def target_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(v.shape for v in self.targets.values())

    def get_features(self, key: str) -> Data:
        return self.features.get(key)

    def get_targets(self, key: str) -> Data:
        return self.targets.get(key)

    def get_tags(self, key: str) -> Data:
        return self.tags.get(key)

    def to_backend(self, backend: Backend) -> Sample:
        return Sample(
            features={k: v.to_backend(backend) for k, v in self.features.items()},
            targets={k: v.to_backend(backend) for k, v in self.targets.items()},
            tags={k: v.to_backend(backend) for k, v in self.tags.items()},
            label=self.label,
            uuid=self.uuid,
        )

    def copy(self) -> Sample:
        return Sample(
            features={k: copy.deepcopy(v) for k, v in self.features.items()},
            targets={k: copy.deepcopy(v) for k, v in self.targets.items()},
            tags={k: copy.deepcopy(v) for k, v in self.tags.items()},
            label=self.label,
            uuid=self.uuid,
        )
