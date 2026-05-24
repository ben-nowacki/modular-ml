"""Package-wide configuration for ModularML."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModularMLSettings:
    """
    Package-wide settings for ModularML.

    Attributes:
        require_unique_labels (bool):
            When True, ExperimentContext enforces unique node labels using the
            ERROR registration policy, regardless of per-context policy.
            Configurable via the ``MODULARML_REQUIRE_UNIQUE_LABELS`` env var
            or :func:`configure`.

    """

    require_unique_labels: bool = False


def _init_defaults() -> ModularMLSettings:
    val = os.getenv("MODULARML_REQUIRE_UNIQUE_LABELS", "").strip().lower()
    return ModularMLSettings(require_unique_labels=val in ("1", "true", "yes"))


settings: ModularMLSettings = _init_defaults()
"""Module-level settings singleton. Mutate directly or use :func:`configure`."""


def configure(*, require_unique_labels: bool | None = None) -> None:
    """
    Programmatically configure package-wide ModularML settings.

    Args:
        require_unique_labels (bool | None, optional):
            When True, all new ExperimentContexts enforce unique node labels
            (ERROR registration policy). Defaults to no change.

    """
    if require_unique_labels is not None:
        settings.require_unique_labels = require_unique_labels
