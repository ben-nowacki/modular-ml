from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modularml.core.references.data_reference import DataReference
from modularml.utils.serialization import SerializableMixin


@dataclass
class SplitterRecord(SerializableMixin):
    """
    Record describing an applied data-splitting operation.

    Description:
        A `SplitterRecord` captures metadata about a single call to a \
        :class:`BaseSplitter` subclass (e.g., :class:`RandomSplitter`, \
        :class:`ConditionSplitter`). It documents where the splitter was \
        applied, whether it targeted a full :class:`FeatureSet` or a \
        :class:`FeatureSetView`, and which configuration parameters were used.

        Each FeatureSet instance maintains an ordered list of `SplitterRecord` \
        entries under its `_split_configs` dictionary, allowing the complete \
        history of applied splits to be reconstructed or replayed deterministically.

    Attributes:
        split_config (dict[str, Any]):
            The configuration dictionary returned by \
            :meth:`BaseSplitter.get_config()`, containing all parameters \
            necessary to reproduce the split operation (e.g., ratios, seed, \
            grouping keys, conditions).

        applied_to (FeatureSetRef):
            A reference to the FeatureSet or FeatureSetView on which this \
            splitter was applied. This includes the FeatureSet label, \
            collection key, and optional split or fold identifiers.

    """

    splitter_state: dict[str, Any]
    applied_to: DataReference

    # ==========================================
    # SerializableMixin
    # ==========================================
    def get_state(self) -> dict:
        """Return full serializable state."""
        return {
            "version": "1.0",
            "splitter_state": self.splitter_state,
            "applied_to": self.applied_to.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Restore SplitterRecord state from a serialized dictionary."""
        version = state.get("version")
        if version != "1.0":
            msg = f"Unsupported SplitterRecord version: {version}"
            raise NotImplementedError(msg)

        self.splitter_state = state["splitter_state"]
        self.applied_to = DataReference.from_state(state["applied_to"])

    @classmethod
    def from_state(cls, state: dict) -> SplitterRecord:
        return cls(
            splitter_state=state["splitter_state"],
            applied_to=DataReference.from_state(state["applied_to"]),
        )
