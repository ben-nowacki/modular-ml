from dataclasses import dataclass
from typing import Any


@dataclass
class LossRecord:
    """
    A container for storing the result of computing a loss value.

    Attributes:
        label (str): Identifier for the loss (e.g., "mse_loss", "triplet_margin").
        value (Any): Computed loss value (typically a backend-specific tensor or scalar).

    """

    value: Any  # raw value output by AppliedLoss.loss_function
    label: str  # AppliedLoss.label
    contributes_to_update: bool = False  # True = loss is a trainable loss, False = auxillary loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "contributes_to_update": self.contributes_to_update,
        }
