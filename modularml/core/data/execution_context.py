"""Execution context for a single batch pass through the model graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modularml.core.training.loss_record import LossCollection

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.batch_view import BatchView
    from modularml.core.references.featureset_reference import FeatureSetReference


@dataclass
class ExecutionContext:
    """
    Execution-time container for a single batch.

    Description:
        Stores all inputs, outputs, losses, and metrics produced during a
        single forward/backward pass through the ModelGraph.

    Attributes:
        phase_label (str): Label identifying the current phase.
        epoch_idx (int): Current epoch index.
        batch_idx (int): Current batch index within the epoch.
        inputs (dict): Pre-materialized :class:`Batch` inputs to head nodes,
            keyed by ``(node_id, upstream_ref)``. Values are concrete tensor
            batches that have already been device-placed by the phase before
            execution begins.
        input_views (dict): Original lazy :class:`BatchView` objects for
            each active (non-masked) head-node binding, keyed by
            ``(node_id, upstream_ref)``. Populated by phases so that loss
            functions that reference specific FeatureSet columns can
            re-materialize with a custom column filter.
        outputs (dict[str, Batch]): Outputs of graph nodes, keyed by node ID.
        losses (LossCollection | None): Losses computed in this batch.

    """

    # Identity
    phase_label: str
    epoch_idx: int
    batch_idx: int

    # Pre-materialized inputs to head nodes (keyed by node ID + upstream ref)
    inputs: dict[tuple[str, FeatureSetReference], Batch] = field(
        default_factory=dict,
    )

    # Original BatchView objects for each active binding (for column-level loss resolution)
    input_views: dict[tuple[str, FeatureSetReference], BatchView] = field(
        default_factory=dict,
    )

    # Outputs of GraphNodes (keyed by node ID)
    outputs: dict[str, Batch] = field(default_factory=dict)

    # Losses computed in this batch
    losses: LossCollection | None = None

    # ================================================
    # Attribute Updating
    # ================================================
    def set_input(
        self,
        *,
        node_id: str,
        upstream: FeatureSetReference,
        batch: Batch,
    ):
        """
        Register a pre-materialized :class:`Batch` for a head node.

        Args:
            node_id (str): Node identifier.
            upstream (FeatureSetReference): Upstream reference for the node.
            batch (Batch): The pre-materialized input :class:`Batch`.

        """
        self.inputs[(node_id, upstream)] = batch

    def set_input_view(
        self,
        *,
        node_id: str,
        upstream: FeatureSetReference,
        batch_view: BatchView,
    ):
        """
        Register the originating :class:`BatchView` for a head node input.

        Args:
            node_id (str): Node identifier.
            upstream (FeatureSetReference): Upstream reference for the node.
            batch_view (BatchView): The lazy :class:`BatchView` that was
                materialized into the corresponding :attr:`inputs` entry.

        """
        self.input_views[(node_id, upstream)] = batch_view

    def set_output(self, *, node_id: str, batch: Batch):
        """
        Sets the tracked outputs for a given node ID.

        Args:
            node_id (str): Node ID to set.
            batch (Batch): Batch data to record.

        """
        if node_id in self.outputs:
            msg = f"Data already set for node: '{node_id}'."
            raise ValueError(msg)
        self.outputs[node_id] = batch

    def add_losses(self, lc: LossCollection):
        """
        Update the tracked losses with this collection.

        Args:
            lc (LossCollection): Loss records to merge.

        """
        if self.losses is None:
            self.losses = lc

        # Combine collections
        else:
            self.losses = LossCollection(
                records=[*lc.values(), *self.losses.values()],
            )
