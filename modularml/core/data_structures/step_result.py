from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modularml.core.data_structures.batch import Batch, BatchOutput
    from modularml.core.loss.loss_result import LossResult


@dataclass
class StepResult:
    """Unified output type for a training or evaluation step."""

    total_loss: float
    total_opt_loss: float = 0.0
    total_non_opt_loss: float = 0.0

    all_loss_results: dict[str, list[LossResult]] = field(default_factory=dict)
    node_outputs: dict[str, Batch | BatchOutput] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_loss": self.total_loss,
            "total_opt_loss": self.total_opt_loss,
            "total_non_opt_loss": self.total_non_opt_loss,
            "all_loss_results": self.all_loss_results,
            "node_outputs": self.node_outputs,
        }
