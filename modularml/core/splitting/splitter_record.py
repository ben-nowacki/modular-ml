from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modularml.core.io.protocols import Configurable
from modularml.core.references.data_reference import DataReference
from modularml.core.splitting.base_splitter import BaseSplitter


@dataclass(frozen=True)
class SplitterRecord(Configurable):
    """Record describing an applied data-splitting operation."""

    splitter: BaseSplitter
    applied_to: DataReference

    def __eq__(self, other):
        if not isinstance(other, SplitterRecord):
            msg = f"Cannot compare equality between SplitterRecord and {type(other)}"
            raise TypeError(msg)

        return (self.splitter == other.splitter) and (self.applied_to == other.applied_to)

    __hash__ = None

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration.

        Note:
            This config does not include the splitter instance, only the
            `splitter.get_config()` dict.

        Returns:
            dict[str, Any]: Configuration used to reconstruct this record.

        """
        return {
            "splitter_config": self.splitter.get_config(),
            "applied_to_config": self.applied_to.get_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SplitterRecord:
        """Reconstructs the record from config."""
        return cls(
            splitter=BaseSplitter.from_config(config["splitter_config"]),
            applied_to=DataReference.from_config(config["applied_to_config"]),
        )
