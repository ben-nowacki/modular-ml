


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from modularml.core.model_graph.loss import LossResult
    from modularml.core.data_structures.batch import Batch, BatchOutput


@dataclass
class StepResult:
    """Unified output type for a training or evaluation step"""
    total_loss: float
    total_opt_loss: float = 0.0
    total_non_opt_loss: float = 0.0
    
    all_loss_results: Dict[str, List["LossResult"]] = field(default_factory=dict)
    stage_outputs: Dict[str, Union["Batch", "BatchOutput"]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": self.total_loss,
            "total_opt_loss": self.total_opt_loss,
            "total_non_opt_loss": self.total_non_opt_loss,
            "all_loss_results": self.all_loss_results,
            "stage_outputs": self.stage_outputs,
        }