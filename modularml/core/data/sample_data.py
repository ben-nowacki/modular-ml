from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_SAMPLE_ID, DOMAIN_TAGS, DOMAIN_TARGETS
from modularml.utils.data.conversion import convert_to_format
from modularml.utils.data.data_format import get_data_format_for_backend
from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from modularml.utils.nn.backend import Backend


class SampleData(Summarizable):
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
        kind: Literal["input", "output"] = "input",
    ):
        self._kind = kind

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

    # ================================================
    # Data access
    # ================================================
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

    @property
    def outputs(self):
        if self._kind != "output":
            raise AttributeError("`outputs` is only defined for SampleData produced by a model.")
        return self.features

    # ================================================
    # Backend casting
    # ================================================
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

    # ================================================
    # Representation
    # ================================================
    @property
    def _primary_domain_name(self) -> str:
        return "outputs" if self._kind == "output" else "features"

    def _summary_rows(self) -> list[tuple]:
        rows: list[tuple] = []

        def _add_domain_row(
            row_label: str,
            domain_data: Any,
        ):
            if domain_data is not None:
                try:
                    shape = str(tuple(domain_data.shape))
                except Exception:  # noqa: BLE001
                    shape = "N/A"

            rows.append((row_label, shape))

        for domain_name, domain_data in self.data.items():
            row_label = domain_name
            if domain_name == "features":
                row_label = self._primary_domain_name
            _add_domain_row(row_label=row_label, domain_data=domain_data)

        return rows

    def __repr__(self) -> str:
        f = self.features.shape if self.features is not None else None
        t = self.targets.shape if self.targets is not None else None
        g = self.tags.shape if self.tags is not None else None
        return f"SampleData({self._primary_domain_name}={f}, targets={t}, tags={g})"


class RoleData(Mapping[str, SampleData], Summarizable):
    """
    Immutable, role-keyed container for SampleData objects.

    Description:
        RoleData is a lightweight convenience wrapper around
        `dict[str, SampleData]` that provides:

        - attribute-style role access
        - role introspection
        - readable summaries
        - a stable return type for model forward passes

        It contains no graph, batch, or sampler semantics.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, SampleData]):
        if not data:
            raise ValueError("RoleData must contain at least one role.")

        for role, sd in data.items():
            if not isinstance(role, str):
                msg = f"Role keys must be str, got {type(role)}"
                raise TypeError(msg)
            if not isinstance(sd, SampleData):
                msg = f"Role '{role}' must map to SampleData, got {type(sd)}"
                raise TypeError(msg)

        self._data = dict(data)

    # ================================================
    # Mapping interface
    # ================================================
    def __getitem__(self, role: str) -> SampleData:
        return self._data[role]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ================================================
    # Role access
    # ================================================
    @property
    def roles(self) -> list[str]:
        """List of available role names."""
        return list(self._data.keys())

    def get_role(self, role: str, default: Any = None) -> SampleData | Any:
        return self._data.get(role, default)

    # ================================================
    # Attribute access
    # ================================================
    def __getattr__(self, name: str) -> SampleData:
        # Called only if attribute not found normally
        if name in self._data:
            return self._data[name]
        msg = f"{self.__class__.__name__} has no role '{name}'. Available roles: {self.roles}"
        raise AttributeError(msg)

    # ================================================
    # Utilities
    # ================================================
    def to_dict(self) -> dict[str, SampleData]:
        """Explicitly unwrap to a plain dict."""
        return dict(self._data)

    def to_backend(self, backend) -> RoleData:
        """
        Cast all SampleData objects to a backend-compatible format.

        Returns:
            RoleData: New RoleData with converted SampleData.

        """
        return RoleData({role: sd.to_backend(backend) or sd for role, sd in self._data.items()})

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        rows: list[tuple] = []

        # Avialable roles
        rows.append(("roles", [(r, "") for r in self.roles]))

        # One row per role with SampleData summary
        for role, sample_data in self._data.items():
            rows.append((role, sample_data._summary_rows()))

        return rows

    def __repr__(self) -> str:
        return f"RoleData(roles={self.roles})"


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
        return f"SampleShapes({self.shapes})"

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
