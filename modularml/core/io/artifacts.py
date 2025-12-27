from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArtifactHeader:
    """
    Metadata describing a saved ModularML artifact for robust round-trip loading.

    Args:
        mml_version (str):
            ModularML version used to save the artifact.
        schema_version (int):
            Artifact schema version for migrations.
        kind (str):
            Kind code (e.g., "fs", "mg") used for naming conventions.
        class_spec (dict[str, Any]):
            Serialized ClassSpec data.

    """

    mml_version: str
    schema_version: int
    kind: str
    class_spec: dict[str, Any]


@dataclass(frozen=True)
class StateSpec:
    """
    Description of how runtime state is stored on disk.

    Args:
        format (str):
            Encoding format (e.g., "json", "pickle", "torch", "npz").
        files (dict[str, str]):
            Logical state names mapped to filenames.

    """

    format: str
    files: dict[str, str]


@dataclass(frozen=True)
class Artifact:
    """
    Full serialized description of an object: header + config + optional state.

    Args:
        header (ArtifactHeader):
            Artifact metadata.
        config (dict[str, Any]):
            Object config needed for instantiation.
        state (StateSpec | None):
            Optional state file mapping for runtime state.

    """

    header: ArtifactHeader
    config: dict[str, Any]
    state: StateSpec | None
