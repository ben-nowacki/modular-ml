from __future__ import annotations

import uuid
from typing import Any

from modularml.core.data.schema_constants import INVALID_LABEL_CHARACTERS
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.utils.representation.summary import format_summary_box
from modularml.utils.serialization.serializable_mixin import SerializableMixin


class ExperimentNode(SerializableMixin):
    """
    Base class for all nodes within an Experiment.

    Each node is identified by a unique `label`.
    """

    def __init__(
        self,
        label: str,
        *,
        node_id: str | None = None,
        register: bool = True,
    ):
        """
        Initialize a ExperimentNode with a name and register.

        Args:
            label (str): Unique identifier for this node.
            node_id (str, optional): Used only for from_state serialization.
            register (bool, optional): Used only for from_state serialization.

        """
        self._node_id: str = node_id or str(uuid.uuid4())
        self._label: str = label

        # Register to context
        if register:
            ExperimentContext.register_experiment_node(self)

    def _validate_label(self, label: str):
        if any(ch in label for ch in INVALID_LABEL_CHARACTERS):
            msg = (
                f"The label contains invalid characters: `{label}`. "
                f"Label cannot contain any of: {list(INVALID_LABEL_CHARACTERS)}"
            )
            raise ValueError(msg)

    @property
    def node_id(self) -> str:
        """Immutable internal identifier."""
        return self._node_id

    @property
    def label(self) -> str:
        """Get or set the unique label for this node."""
        return self._label

    @label.setter
    def label(self, new_label: str):
        """Get or set the unique label for this node."""
        self._validate_label(label=new_label)
        ExperimentContext.update_node_label(self, new_label)
        self._label = new_label

    # ================================================
    # Representation
    # ================================================
    def summary(self, max_width: int = 88) -> str:
        rows = [
            ("label", self.label),
            ("node_id", self.node_id),
        ]

        return format_summary_box(
            title=self.__class__.__name__,
            rows=rows,
            max_width=max_width,
        )

    def __repr__(self):
        return f"ExperimentNode(label='{self.label}')"

    # ================================================
    # Serialization
    # ================================================
    def get_state(self) -> dict[str, Any]:
        """Base serialization for all ExperimentNodes."""
        return {
            "_target": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "node_id": self.node_id,
            "label": self.label,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore identity and label in-place.

        Important:
        - Must NOT register automatically
        - Registration handled explicitly by caller

        """
        self._node_id = state["node_id"]
        self._label = state["label"]
        self._validate_label(self._label)
