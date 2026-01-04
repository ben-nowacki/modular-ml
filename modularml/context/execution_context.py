from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import RoleData

# TODO:


@dataclass
class ExecutionContext:
    """Container for all data produced and consumed during a single ModelGraph execution step."""

    # Head node inputs data (keyed by the node that takes in the data)
    # Multi heads may consume the same batch, therefore key can be a list of node_ids
    # node_id(s) -> Batch
    inputs: dict[str | Sequence[str], Batch]

    # ModelNode tensor outputs: node_id -> RoleData
    outputs: dict[str, RoleData]

    # # Loss outputs keyed by node_id
    # node_losses: dict[str, LossRecords]

    # # Optional metadata
    # step: int | None = None
    # uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
