from abc import ABC, abstractmethod
from typing import Any

from modularml.core.data_structures.sample import Sample


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, samples: list[Sample]) -> dict[str, list[str]]:
        """Returns a dictionary mapping subset names to `Sample.uuid`."""
        raise NotImplementedError

    @abstractmethod
    def get_config(
        self,
    ) -> dict[str, Any]:
        """Returns a configuration to reproduce identical split configurations."""

    @classmethod
    @abstractmethod
    def from_config(cls, config) -> "BaseSplitter":
        """Instantiates a BaseSplitter with a config."""
