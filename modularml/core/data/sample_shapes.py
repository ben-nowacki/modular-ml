from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_TAGS, DOMAIN_TARGETS


class SampleShapes:
    """
    Shape specification for all schema domains with tensor-like data.

    Description:
        Defines the per-sample tensor shapes of all schema domains \
        (features, targets, tags) for one node in the model graph. \
        Shapes are stored without the batch dimension, since batch size \
        may vary at runtime and is not part of the static schema.

        The same shape applies to all roles associated with the node.

    Args:
        shapes (dict[str, tuple[int, ...]]):
            Dictionary mapping each schema domain name \
            (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS) to a shape \
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
                self.shapes[DOMAIN_FEATURES] = features_shape
            if targets_shape is not None:
                self.shapes[DOMAIN_TARGETS] = targets_shape
            if tags_shape is not None:
                self.shapes[DOMAIN_TAGS] = tags_shape

    def __getitem__(self, domain: str) -> tuple[int, ...]:
        """The shape tuple corresponding to a given schema domain."""
        return self.shapes[domain]

    def __repr__(self):
        return self.shapes

    @property
    def features_shape(self) -> tuple[int, ...]:
        """Shape tuple for the DOMAIN_FEATURES."""
        return self.shapes[DOMAIN_FEATURES]

    @property
    def targets_shape(self) -> tuple[int, ...]:
        """Shape tuple for the DOMAIN_TARGETS domain."""
        return self.shapes[DOMAIN_TARGETS]

    @property
    def tags_shape(self) -> tuple[int, ...]:
        """Shape tuple for the DOMAIN_TAGS domain."""
        return self.shapes[DOMAIN_TAGS]
