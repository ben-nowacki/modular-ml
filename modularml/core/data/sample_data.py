from typing import Any

from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_SAMPLE_ID, DOMAIN_TAGS, DOMAIN_TARGETS
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import get_data_format_for_backend
from modularml.utils.nn.backend import Backend


class SampleData:
    """
    Tensor-like data container for a single role across all schema domains.

    Description:
        Represents the runtime data (features, targets, and tags) associated with \
        a single role within a Batch. Each SampleData instance stores tensors or \
        array-like objects under standardized domain keys defined in \
        `SampleSchema`, including:
          - DOMAIN_FEATURES
          - DOMAIN_TARGETS
          - DOMAIN_TAGS

        This structure allows consistent access to domain data using either \
        standardized constants or convenience properties.

    Args:
        data (dict[str, Any]):
            Dictionary mapping schema domain names \
            (DOMAIN_FEATURES, DOMAIN_TARGETS, DOMAIN_TAGS) to tensor-like values.
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
                self.data[DOMAIN_SAMPLE_ID] = sample_uuids
            if features is not None:
                self.data[DOMAIN_FEATURES] = features
            if targets is not None:
                self.data[DOMAIN_TARGETS] = targets
            if tags is not None:
                self.data[DOMAIN_TAGS] = tags

    @property
    def sample_uuids(self):
        """Tensor-like data stored under DOMAIN_SAMPLE_ID."""
        return self.data.get(DOMAIN_SAMPLE_ID)

    @property
    def features(self):
        """Tensor-like data stored under DOMAIN_FEATURES."""
        return self.data.get(DOMAIN_FEATURES)

    @property
    def targets(self):
        """Tensor-like data stored under DOMAIN_TARGETS."""
        return self.data.get(DOMAIN_TARGETS)

    @property
    def tags(self):
        """Tensor-like data stored under DOMAIN_TAGS."""
        return self.data.get(DOMAIN_TAGS)

    def __repr__(self) -> str:
        return f"SampleData(features={self.features.shape}, targets={self.targets.shape}, tags={self.tags.shape})"

    def __str__(self):
        return self.__repr__()

    def summary(self, *, include_none: bool = False) -> str:
        """
        Return a human-readable multi-line summary of the SampleData contents.

        Description:
            Shows the available schema domains (features, targets, tags), the
            tensor shapes for each, and any missing domains if `include_none=True`.

        Returns:
            str: Formatted summary block.

        """
        rows = []

        for key, val in self.data.items():
            if val is not None or include_none:
                # Get shape if tensor-like
                try:
                    shape = tuple(val.shape)
                except Exception:  # noqa: BLE001
                    shape = "N/A"
                rows.append(f"{key:<12}: {shape}")

        # If empty, add placeholder
        if not rows:
            rows.append("(no data)")

        width = max(len(r) for r in rows)
        title = f"{self.__class__.__name__}"
        border = "─" * width

        body = "\n".join(rows)
        return f"┌─ {title} {'─' * (width - len(title) - 3)}┐\n{body}\n└{border}┘"

    def to_backend(self, backend: Backend):
        """
        Casts the data to a DataFormat compatible with specified backend.

        Description:
            Performs an in-place data casting of the underlying tensors.
            Reformats the features and targets, but not the tags.

        """
        self.data[DOMAIN_FEATURES] = convert_to_format(
            data=self.data[DOMAIN_FEATURES],
            fmt=get_data_format_for_backend(backend=backend),
        )
        self.data[DOMAIN_TARGETS] = convert_to_format(
            data=self.data[DOMAIN_TARGETS],
            fmt=get_data_format_for_backend(backend=backend),
        )
