from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.data_structures.sample import Sample
from modularml.utils.backend import Backend
from modularml.utils.data_conversion import convert_to_format
from modularml.utils.data_format import DataFormat

if TYPE_CHECKING:
    from modularml.core.graph.shape_spec import ShapeSpec


class SampleCollection:
    """
    A lightweight container for a list of Sample instances.

    Attributes:
        samples (list[Sample]): A list of Sample instances.

    """

    def __init__(self, samples: list[Sample]):
        if not all(isinstance(s, Sample) for s in samples):
            raise TypeError("All elements in `samples` must be of type Sample.")
        if len(samples) == 0:
            raise ValueError("SampleCollection must contain at least one Sample.")

        self.samples: list[Sample] = samples
        self._uuid_sample_map: dict[str, Sample] = {s.uuid: s for s in samples}
        self._label_sample_map: dict[str, Sample] = {s.label: s for s in samples}

        # Enforce same shapes across samples
        f_shapes = list({s.feature_shape_spec for s in samples})
        if len(f_shapes) != 1:
            msg = f"Inconsistent SampleCollection feature shapes: {f_shapes}."
            raise ValueError(msg)
        t_shapes = list({s.target_shape_spec for s in samples})
        if len(t_shapes) != 1:
            msg = f"Inconsistent SampleCollection target shapes: {t_shapes}."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def get_sample_with_uuid(self, uuid: str) -> Sample:
        return self._uuid_sample_map[uuid]

    def get_samples_with_uuid(self, uuids: list[str]) -> SampleCollection:
        return SampleCollection(samples=[self.get_sample_with_uuid(x) for x in uuids])

    def get_sample_with_label(self, label: str) -> Sample:
        return self._label_sample_map[label]

    def get_samples_with_label(self, labels: list[str]) -> SampleCollection:
        return SampleCollection(samples=[self.get_sample_with_label(x) for x in labels])

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"SampleCollection(n_samples={len(self.samples)})"

    @property
    def feature_keys(self) -> list[str]:
        return self.samples[0].feature_keys

    @property
    def target_keys(self) -> list[str]:
        return self.samples[0].target_keys

    @property
    def tag_keys(self) -> list[str]:
        return self.samples[0].tag_keys

    @property
    def sample_uuids(self) -> list[str]:
        return [s.uuid for s in self.samples]

    @property
    def sample_labels(self) -> list[str]:
        return [s.label for s in self.samples]

    @property
    def feature_shape_spec(self) -> ShapeSpec:
        return self.samples[0].feature_shape_spec

    @property
    def target_shape_spec(self) -> ShapeSpec:
        return self.samples[0].target_shape_spec

    def get_feature_shape(self, key: str) -> tuple[int, ...]:
        return self.feature_shape_spec.get(key)

    def get_target_shape(self, key: str) -> tuple[int, ...]:
        return self.target_shape_spec.get(key)

    def get_all_features(self, fmt: str | DataFormat = DataFormat.DICT_NUMPY) -> Any:
        """Returns all features across all samples in the specified format."""
        # Map Data instances to each feature key
        raw = {k: [s.features[k] for s in self.samples] for k in self.feature_keys}

        # Convert to specified format
        return convert_to_format(data=raw, fmt=fmt)

    def get_all_targets(self, fmt: str | DataFormat = DataFormat.DICT_NUMPY) -> Any:
        """Returns all targets across all samples in the specified format."""
        # Map Data instances to each target key
        raw = {k: [s.targets[k] for s in self.samples] for k in self.target_keys}

        # Convert to specified format
        return convert_to_format(data=raw, fmt=fmt)

    def get_all_tags(self, fmt: str | DataFormat = DataFormat.DICT_NUMPY) -> Any:
        """
        Returns all tags across all samples in the specified format.

        Each tag will be returned as a list of values across samples.
        The format argument controls the output structure.
        """
        # Map Data instances to each target key
        raw = {k: [s.tags[k] for s in self.samples] for k in self.tag_keys}

        # Convert to specified format
        return convert_to_format(data=raw, fmt=fmt)

    def to_backend(self, backend: str | Backend) -> SampleCollection:
        """Returns a new SampleCollection with all Data objects converted to the specified backend."""
        if isinstance(backend, str):
            backend = Backend(backend)
        converted = [s.to_backend(backend) for s in self.samples]
        return SampleCollection(samples=converted)

    def copy(self) -> SampleCollection:
        """
        Create a deep copy of the SampleCollection, including all contained Sample instances.

        Returns:
            SampleCollection: A new SampleCollection with identical Samples.

        """
        copied_samples = [s.copy() for s in self.samples]
        return SampleCollection(samples=copied_samples)
