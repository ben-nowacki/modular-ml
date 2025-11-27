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
SHAPE_SUFFIX = "shape"
DTYPE_SUFFIX = "dtype"

# Reserved domain column names
FEATURES_COLUMN = "features"
TARGETS_COLUMN = "targets"
TAGS_COLUMN = "tags"
SAMPLE_ID_COLUMN = "sample_id"

# Reserved column variants
RAW_VARIANT = "raw"
TRANSFORMED_VARIANT = "transformed"


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

        The schema acts as the contract that ensures consistent column names, \
        data types, and separation across these domains.

        Every Arrow table that follows this schema must also contain a \
        global identifier column, `SAMPLE_ID_COLUMN`, storing a unique \
        per-row ID string (UUID or hash).

    """

    # Mapping: domain -> column name -> column variant -> dtype
    features: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)
    targets: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)
    tags: Mapping[str, Mapping[str, pa.DataType]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates and normalizes schema mappings after initialization.

        Ensures that:
          1. All mappings are dicts (not arbitrary Mappings).
          2. No duplicate column names appear across domains.
          3. Columns names do not contains invalid characters.
        """
        # Instantiate dicts
        self.features = {k: dict(v) for k, v in dict(self.features).items()}
        self.targets = {k: dict(v) for k, v in dict(self.targets).items()}
        self.tags = {k: dict(v) for k, v in dict(self.tags).items()}

        # Detect duplicate column names across domains
        duplicates = (
            (set(self.features) & set(self.targets))
            | (set(self.features) & set(self.tags))
            | (set(self.targets) & set(self.tags))
        )
        if duplicates:
            msg = f"Schema keys must be unique across features/targets/tags. Duplicated keys: {sorted(duplicates)}"
            raise ValueError(msg)

        # Check for invalid characters and reserved names
        all_keys = set(self.features.keys()) | set(self.targets.keys()) | set(self.tags.keys())
        validate_str_list(list(all_keys))

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
        domain = domain.lower()
        if domain == FEATURES_COLUMN:
            return list(self.features.keys())
        if domain == TARGETS_COLUMN:
            return list(self.targets.keys())
        if domain == TAGS_COLUMN:
            return list(self.tags.keys())
        msg = f"Unknown domain '{domain}'. Expected one of: {FEATURES_COLUMN}, {TARGETS_COLUMN}, {TAGS_COLUMN}"
        raise ValueError(msg)

    def domain_types(self, domain: str) -> dict[str, dict[str, pa.DataType]]:
        """
        Return the {column_name: DataType} mapping for a given domain.

        Args:
            domain: Domain name ("features", "targets", "tags").

        Returns:
            Mapping of column names to variants to Arrow data types.

        """
        domain = domain.lower()
        if domain == FEATURES_COLUMN:
            return self.features
        if domain == TARGETS_COLUMN:
            return self.targets
        if domain == TAGS_COLUMN:
            return self.tags
        msg = f"Unknown domain '{domain}'. Expected one of: {FEATURES_COLUMN}, {TARGETS_COLUMN}, {TAGS_COLUMN}"
        raise ValueError(msg)

    def variant_keys(self, domain: str, key: str) -> list[str]:
        """
        Return available variants for a given column key.

        Args:
            domain (str): Domain name ("features", "targets", "tags").
            key (str): Column name in specified domain.

        Returns:
            list[str]: Variant names

        """
        dom_types = self.domain_types(domain)
        if key not in dom_types:
            msg = f"Column '{key}' not found in domain '{domain}'"
            raise KeyError(msg)
        return list(dom_types[key].keys())

    def variant_types(self, domain: str, key: str) -> dict[str, pa.DataType]:
        """
        Return the {variant_name: DataType} mapping for a given domain and column.

        Args:
            domain (str): Domain name ("features", "targets", "tags").
            key (str): Column name in specified domain.

        Returns:
            Mapping of variants to Arrow data types.

        """
        dom_types = self.domain_types(domain)
        if key not in dom_types:
            msg = f"Column '{key}' not found in domain '{domain}'"
            raise KeyError(msg)
        return dom_types[key]

    # =================================================================
    # Flat schema inference
    # =================================================================
    @classmethod
    def from_table(cls, table: pa.Table) -> SampleSchema:
        """
        Infer SampleSchema from a flat Arrow table.

        Expected column naming:
            "<domain>.<key>.<variant>"

        Example:
            features.voltage.raw
            features.voltage.transformed
            targets.soh.raw
            tags.cell_id.raw
            sample_id

        """
        features: dict[str, dict[str, pa.DataType]] = {}
        targets: dict[str, dict[str, pa.DataType]] = {}
        tags: dict[str, dict[str, pa.DataType]] = {}

        for col in table.schema.names:
            if col == SAMPLE_ID_COLUMN:
                continue

            parts = col.split(".")
            if len(parts) != 3:
                msg = f"Invalid column '{col}'. Expected '<domain>.<key>.<variant>' format."
                raise ValueError(msg)

            domain, key, variant = parts
            dtype = table.schema.field(col).type

            if domain == FEATURES_COLUMN:
                features.setdefault(key, {})[variant] = dtype
            elif domain == TARGETS_COLUMN:
                targets.setdefault(key, {})[variant] = dtype
            elif domain == TAGS_COLUMN:
                tags.setdefault(key, {})[variant] = dtype
            else:
                msg = f"Unknown domain '{domain}' in column '{col}'"
                raise ValueError(msg)

        return cls(features=features, targets=targets, tags=tags)


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

    sample_ids = pa.array([str(uuid.uuid4()) for _ in range(table.num_rows)], type=pa.string())
    return table.append_column(SAMPLE_ID_COLUMN, sample_ids)


def validate_str_list(keys: list[str]):
    """
    Validate a list of string keys for naming consistency and safety.

    Description:
        Ensures that all provided keys meet naming requirements for use in
        FeatureSet, SampleCollection, or related schema contexts. This function
        enforces:
          - Uniqueness of all keys.
          - Absence of invalid characters (see `INVALID_LABEL_CHARACTERS`).
          - Exclusion of reserved schema keywords (e.g., `SAMPLE_ID_COLUMN`,
            `FEATURES_COLUMN`, etc.).
          - Exclusion of internal metadata prefixes/postfixes used in column
            naming conventions.

    Args:
        keys (list[str]):
            List of strings to validate.

    Raises:
        ValueError:
            If any of the following conditions occur:
              - Duplicate keys are detected.
              - Keys contain invalid characters.
              - Keys use reserved schema names or internal label conventions.

    Example:
        ```python
        validate_str_list(["cell_id", "cycle_number", "soh"])  # OK
        validate_str_list(["cell.id", "cycle_number"])  # Raises ValueError
        ```

    """
    # Detect duplicate elements in keys
    if len(set(keys)) != len(keys):
        msg = "Duplicate elements exist in `keys`."
        raise ValueError(msg)

    # Check invalid characters
    invalid_chars = tuple(INVALID_LABEL_CHARACTERS)
    invalid_keys: list[str] = []
    for k in keys:
        if any(k.find(ch) != -1 for ch in invalid_chars):
            invalid_keys.append(k)
    if invalid_keys:
        msg = (
            f"The following keys contain invalid characters: {', '.join(invalid_keys)}. "
            f"Keys cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
        )
        raise ValueError(msg)

    # Ensure reserved names are not used
    for res_key in (
        SAMPLE_ID_COLUMN,
        FEATURES_COLUMN,
        TARGETS_COLUMN,
        TAGS_COLUMN,
        RAW_VARIANT,
        TRANSFORMED_VARIANT,
        METADATA_PREFIX,
        DTYPE_SUFFIX,
        SHAPE_SUFFIX,
    ):
        if res_key in keys:
            msg = f"`{res_key}` is a reserved keyword and cannot be used."
            raise ValueError(msg)
