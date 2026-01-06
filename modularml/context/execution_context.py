from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modularml.core.data.batch import Batch
    from modularml.core.data.batch_view import BatchView
    from modularml.core.training.loss_record import LossCollection


@dataclass
class ExecutionContext:
    """Container for all data produced and consumed during a single ModelGraph execution step."""

    # Head node inputs data (keyed by the node that takes in the data)
    # Multi heads may consume the same batch, therefore key can be a list of node_ids
    # node_id(s) -> Batch
    # We also record BatchView in case losses use non-materialized data
    input_views: dict[str | Sequence[str], BatchView]
    input_batches: dict[str | Sequence[str], Batch]

    # ModelNode tensor outputs: node_id -> Batch
    model_outputs: dict[str, Batch] | None = None

    # Loss outputs keyed by node_id
    model_losses: dict[str, LossCollection] | None = None

    # # Optional metadata
    # step: int | None = None
    # uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
