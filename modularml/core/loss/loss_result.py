from dataclasses import dataclass
from typing import Any


@dataclass
class LossResult:
    """
    A container for storing the result of computing a loss value.

    Attributes:
        label (str): Identifier for the loss (e.g., "mse_loss", "triplet_margin").
        value (Any): Computed loss value (typically a backend-specific tensor or scalar).

    """

    label: str
    value: Any
