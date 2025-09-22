from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modularml.core.data_structures.batch import Batch, BatchOutput
    from modularml.core.loss.loss_collection import LossCollection


@dataclass
class StepResult:
    """Unified output type for a training or evaluation step."""

    losses: LossCollection
    node_outputs: dict[str, Batch | BatchOutput] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "losses": self.losses.as_dict(),
            "node_outputs": self.node_outputs,
        }
