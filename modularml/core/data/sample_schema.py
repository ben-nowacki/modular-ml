from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Mapping

# =====================================================================
# ModularML Sample Schema Constants
# =====================================================================
SCHEMA_VERSION = "1.0.0"
INVALID_LABEL_CHARACTERS: set[str] = {".", " ", "/", "\\", ":"}

# Metadata keys / prefixes embedded in Arrow tables
METADATA_PREFIX = "modularml"
METADATA_SCHEMA_VERSION_KEY = f"{METADATA_PREFIX}.version"
SHAPE_POSTFIX = "shape"
DTYPE_POSTFIX = "dtype"

# Canonical domain column names
FEATURES_COLUMN = "features"
TARGETS_COLUMN = "targets"
TAGS_COLUMN = "tags"
SAMPLE_ID_COLUMN = "sample_id"


# =====================================================================
# SampleSchema
# =====================================================================
@dataclass
class SampleSchema:
    """
    Defines and validates the schema of a FeatureSet.

    Description:
        Each FeatureSet is organized into three structured domains:
          - **features**: model inputs (e.g., voltage, current)
          - **targets**: supervised outputs (e.g., SOH, capacity)
          - **tags**: metadata or identifiers (e.g., cell_id, SOC)

        The schema acts as the contract that ensures consistent column names,
        data types, and separation across these domains.

        Every Arrow table that follows this schema must also contain a
        global identifier column, `SAMPLE_ID_COLUMN`, storing a unique
        per-row ID string (UUID or hash).

    """

    # Domain mapping of column name to pyarrow.DataType
    features: Mapping[str, pa.DataType] = field(default_factory=dict)
    targets: Mapping[str, pa.DataType] = field(default_factory=dict)
    tags: Mapping[str, pa.DataType] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates and normalizes schema mappings after initialization.

        Ensures that:
          1. All mappings are dicts (not arbitrary Mappings).
          2. No duplicate column names appear across domains.
          3. Columns names do not contains invalid characters.
        """
        # Normalize to concrete dicts (so `.keys()` and `.items()` are stable)
        self.features = dict(self.features)
        self.targets = dict(self.targets)
        self.tags = dict(self.tags)

        # Detect duplicate column names across domains
        duplicates = (
            (set(self.features) & set(self.targets))
            | (set(self.features) & set(self.tags))
            | (set(self.targets) & set(self.tags))
        )
        if duplicates:
            msg = f"Schema keys must be unique across features/targets/tags. Duplicated keys: {sorted(duplicates)}"
            raise ValueError(msg)

        # Check for invalid characters
        invalid_chars = tuple(INVALID_LABEL_CHARACTERS)
        invalid_keys: list[str] = []
        for mapping in (self.features, self.targets, self.tags):
            for name in mapping:
                if any(name.find(ch) != -1 for ch in invalid_chars):
                    invalid_keys.append(name)
        if invalid_keys:
            msg = (
                f"The following keys contain invalid characters: {', '.join(invalid_keys)}. "
                f"Keys cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
            )
            raise ValueError(msg)

        # Ensure reserved name not used
        for domain in (self.features, self.targets, self.tags):
            if SAMPLE_ID_COLUMN in domain:
                msg = f"`{SAMPLE_ID_COLUMN}` is a reserved column name and cannot appear in schema domains."
                raise ValueError(msg)

    # =================================================================
    # Domain utility methods
    # =================================================================
    def domain_keys(self, domain: str) -> list[str]:
        """
        Return the list of column names for a given domain.

        Args:
            domain: One of {"features", "targets", "tags"}.

        Returns:
            A list of string column names in that domain.

        Raises:
            ValueError: if the domain name is invalid.

        """
        if domain == FEATURES_COLUMN:
            return list(self.features.keys())
        if domain == TARGETS_COLUMN:
            return list(self.targets.keys())
        if domain == TAGS_COLUMN:
            return list(self.tags.keys())
        msg = f"Unknown domain '{domain}'. Expected one of: {FEATURES_COLUMN}, {TARGETS_COLUMN}, {TAGS_COLUMN}"
        raise ValueError(msg)

    def domain_types(self, domain: str) -> dict[str, pa.DataType]:
        """
        Return the {column_name: DataType} mapping for a given domain.

        Args:
            domain: Domain name ("features", "targets", "tags").

        Returns:
            Mapping of column names to Arrow data types.

        """
        if domain == FEATURES_COLUMN:
            return self.features
        if domain == TARGETS_COLUMN:
            return self.targets
        if domain == TAGS_COLUMN:
            return self.tags
        msg = f"Unknown domain '{domain}'. Expected one of: {FEATURES_COLUMN}, {TARGETS_COLUMN}, {TAGS_COLUMN}"
        raise ValueError(msg)

    def struct_type(self, domain: str) -> pa.StructType:
        """
        Build a `pyarrow.StructType` for a given domain.

        This allows a FeatureSet to represent each domain
        as a single structured Arrow column, e.g.:

        ```
        features: struct<voltage: list<float32>, current: list<float32>>
        ```

        Args:
            domain: One of {"features", "targets", "tags"}.

        Returns:
            pa.StructType corresponding to that domain.

        """
        domain_types = self.domain_types(domain)
        fields = [pa.field(name, dtype) for name, dtype in domain_types.items()]
        return pa.struct(fields)

    # =================================================================
    # Constructors and serialization
    # =================================================================
    @classmethod
    def from_table(cls, table: pa.Table) -> SampleSchema:
        """
        Infer a SampleSchema from a pyarrow.Table.

        Args:
            table: Arrow table containing 'features', 'targets', and 'tags' StructArrays.

        Returns:
            A SampleSchema describing each domain's fields and data types.

        """

        def _extract_struct_type(domain: str) -> dict[str, pa.DataType]:
            if domain not in table.column_names:
                return {}
            struct_type = table[domain].type
            if not isinstance(struct_type, pa.StructType):
                msg = f"Column '{domain}' must be a StructType, not {struct_type}."
                raise TypeError(msg)
            return {fld.name: fld.type for fld in struct_type}

        return cls(
            features=_extract_struct_type(FEATURES_COLUMN),
            targets=_extract_struct_type(TARGETS_COLUMN),
            tags=_extract_struct_type(TAGS_COLUMN),
        )


def ensure_sample_id_column(table: pa.Table) -> pa.Table:
    """
    Ensure that the Arrow table contains a unique SAMPLE_ID_COLUMN.

    Description:
        - If the column exists, validates that it is of type string.
        - If it does not exist, generates a new UUID string for each row.
        - This column is used for traceability across subsets and views.

    Args:
        table (pa.Table): Input Arrow table.

    Returns:
        pa.Table: A table guaranteed to contain a valid SAMPLE_ID_COLUMN.

    """
    if SAMPLE_ID_COLUMN in table.column_names:
        col = table[SAMPLE_ID_COLUMN]
        if not pa.types.is_string(col.type):
            msg = f"'{SAMPLE_ID_COLUMN}' column must be of type string, got {col.type}."
            raise TypeError(msg)
        return table

    n = table.num_rows
    sample_ids = pa.array([str(uuid.uuid4()) for _ in range(n)], type=pa.string())
    return table.append_column(SAMPLE_ID_COLUMN, sample_ids)
