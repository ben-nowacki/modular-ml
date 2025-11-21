import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modularml.core.data.sample_schema import FEATURES_COLUMN, SAMPLE_ID_COLUMN, TAGS_COLUMN, TARGETS_COLUMN


class RoleData:
    """
    Tensor-like data container for a single role across all schema domains.

    Description:
        Represents the runtime data (features, targets, and tags) associated with \
        a single role within a Batch. Each RoleData instance stores tensors or \
        array-like objects under standardized domain keys defined in \
        `SampleSchema`, including:
          - FEATURES_COLUMN
          - TARGETS_COLUMN
          - TAGS_COLUMN

        This structure allows consistent access to domain data using either \
        standardized constants or convenience properties.

    Args:
        data (dict[str, Any]):
            Dictionary mapping schema domain names \
            (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN) to tensor-like values.
        features:
            Tensor-like object representing feature data.
        targets:
            Tensor-like object representing target data.
        tags:
            Tensor-like object representing tag data.

    """

    __slots__ = ("data",)
    data: dict[str, Any]

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        sample_uuids: Any = None,
        features: Any = None,
        targets: Any = None,
        tags: Any = None,
    ):
        if data is not None:
            # Use provided dictionary directly
            self.data = dict(data)
        else:
            # Construct standardized mapping
            self.data = {}
            if sample_uuids is not None:
                self.data[SAMPLE_ID_COLUMN] = sample_uuids
            if features is not None:
                self.data[FEATURES_COLUMN] = features
            if targets is not None:
                self.data[TARGETS_COLUMN] = targets
            if tags is not None:
                self.data[TAGS_COLUMN] = tags

    @property
    def sample_uuids(self):
        """Tensor-like data stored under SAMPLE_ID_COLUMN."""
        return self.data.get(SAMPLE_ID_COLUMN)

    @property
    def features(self):
        """Tensor-like data stored under FEATURES_COLUMN."""
        return self.data.get(FEATURES_COLUMN)

    @property
    def targets(self):
        """Tensor-like data stored under TARGETS_COLUMN."""
        return self.data.get(TARGETS_COLUMN)

    @property
    def tags(self):
        """Tensor-like data stored under TAGS_COLUMN."""
        return self.data.get(TAGS_COLUMN)

    def __repr__(self) -> str:
        return f"RoleData(features={self.features.shape}, targets={self.targets.shape}, tags={self.tags.shape})"

    def __str__(self):
        return self.__repr__()


class NodeShapes:
    """
    Shape specification for all schema domains within a single graph node.

    Description:
        Defines the per-sample tensor shapes of all schema domains \
        (features, targets, tags) for one node in the model graph. \
        Shapes are stored without the batch dimension, since batch size \
        may vary at runtime and is not part of the static schema.

        The same shape applies to all roles associated with the node.

    Args:
        shapes (dict[str, tuple[int, ...]]):
            Dictionary mapping each schema domain name \
            (FEATURES_COLUMN, TARGETS_COLUMN, TAGS_COLUMN) to a shape \
            tuple (excluding batch dimension).
        features_shape (tuple[int, ...]]):
            Tuple of ints representing feature shape.
        targets_shape (tuple[int, ...]]):
            Tuple of ints representing target shape.
        tags_shape (tuple[int, ...]]):
            Tuple of ints representing tag shape.

    """

    __slot__ = ("shapes",)
    shapes: dict[str, tuple[int, ...]]

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]] | None = None,
        *,
        features_shape: tuple[int, ...] | None = None,
        targets_shape: tuple[int, ...] | None = None,
        tags_shape: tuple[int, ...] | None = None,
    ):
        if shapes is not None:
            # Use provided dictionary directly
            self.shapes = dict(shapes)
        else:
            # Construct standardized mapping
            self.shapes = {}
            if features_shape is not None:
                self.shapes[FEATURES_COLUMN] = features_shape
            if targets_shape is not None:
                self.shapes[TARGETS_COLUMN] = targets_shape
            if tags_shape is not None:
                self.shapes[TAGS_COLUMN] = tags_shape

    def __getitem__(self, domain: str) -> tuple[int, ...]:
        """The shape tuple corresponding to a given schema domain."""
        return self.shapes[domain]

    def __repr__(self):
        return f"ShapeSpec({self.shapes})"

    @property
    def features_shape(self) -> tuple[int, ...]:
        """Shape tuple for the FEATURES_COLUMN domain."""
        return self.shapes[FEATURES_COLUMN]

    @property
    def targets_shape(self) -> tuple[int, ...]:
        """Shape tuple for the TARGETS_COLUMN domain."""
        return self.shapes[TARGETS_COLUMN]

    @property
    def tags_shape(self) -> tuple[int, ...]:
        """Shape tuple for the TAGS_COLUMN domain."""
        return self.shapes[TAGS_COLUMN]


@dataclass
class Batch:
    """
    Materialized data container for one forward or training pass through the graph.

    Description:
        A Batch represents a collection of role-specific data derived from one or \
        more FeatureSets, organized per model-graph node. Each node's output data \
        is stored as a mapping of roles to RoleData objects, containing tensors \
        for the standardized schema domains (features, targets, tags).

        The Batch also records optional sample weights and per-node output shapes.

    Args:
        outputs:
            Dictionary mapping node labels to role mappings, where each role maps \
            to a RoleData object. Structure: \
                node_label -> role -> RoleData(domain -> tensor-like data)

        role_sample_weights:
            Optional mapping of role names to per-sample weights \
            (array-like of shape (batch_size,)).

        shapes:
            Optional mapping of node labels to NodeShapes objects, describing the \
            tensor shapes for each schema domain (excluding batch dimension).

        uuid:
            Unique identifier automatically generated for tracking/logging this batch.

    """

    # Outputs of each node
    # Initiallized with FeatureSet sample during batch creation
    # Keyed as follows: node_label -> role -> RoleData(feature/target/tag -> tensor-like data)
    outputs: dict[str, dict[str, RoleData]]

    # Weights of each sample in each role
    # Initiallize during batch creation (and then immutable?)
    # Keyed as follows: role -> np.ndarray with shape (batch_size,)
    role_sample_weights: dict[str, np.ndarray]

    # Shape of outputs
    # Keyed as follows: node_label -> NodeShapes(feature/target/tag -> tuple[int, ...])
    shapes: dict[str, NodeShapes]

    # Unique ID for logging/tracking batches
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    # TODO: Other data that might be good to track:
    # The FeatureSet/Subset/view that was used to instantiate this batch
    # The sampler used in conjunction with above to instantiate this batch
    # This might be better to track in BatchContext
