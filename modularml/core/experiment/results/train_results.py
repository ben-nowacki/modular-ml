"""Results container for training phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from modularml.core.experiment.results.phase_results import PhaseResults

if TYPE_CHECKING:
    from modularml.callbacks.early_stopping import EarlyStoppingResult


@dataclass
class TrainResults(PhaseResults):
    """
    Results container for a training phase.

    Description:
        TrainResults wraps the outputs of a TrainPhase, which executes
        multiple epochs with multiple batches per epoch. This class provides:

        - Access to training data keyed by epoch and batches
        - Direct access to validation losses/tensors from evaluation callbacks
        - Loss aggregation per epoch

        Validation callbacks (kind="evaluation") are automatically detected
        and their results exposed through dedicated accessors.

    Attributes:
        label (str): Phase label.
        _execution (list[ExecutionContext]): Ordered execution contexts.
        _callbacks (list[CallbackResult]): Recorded callback outputs.
        _metrics (MetricStore): Stored scalar metrics.
        _series_cache (dict[tuple, Any]): Cache of memoized AxisSeries queries.

    """

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        n_epochs = self.n_epochs if self._execution else 0
        return f"TrainResults(label='{self.label}', epochs={n_epochs})"

    # ================================================
    # Properties
    # ================================================
    @property
    def epoch_indices(self) -> list[int]:
        """
        Sorted list of recorded epoch indices.

        Returns:
            list[int]: Epoch indices in ascending order.

        """
        epoch_vals = self.execution_contexts().axis_values("epoch")
        return sorted(int(e) for e in epoch_vals)

    @property
    def n_epochs(self) -> int:
        """
        The number of epochs executed during training.

        Returns:
            int: Total number of recorded epochs.

        """
        return len(self.epoch_indices)

    # ================================================
    # EarlyStopping Convenience
    # ================================================
    @property
    def early_stopping_result(self) -> EarlyStoppingResult | None:
        """
        Return the EarlyStoppingResult recorded at phase end, if present.

        Returns:
            EarlyStoppingResult | None:
                The result emitted by an :class:`EarlyStopping` callback,
                or ``None`` if no such callback was attached to this phase.

        """
        values = self.callbacks(kind="early_stopping").values()
        return values[0] if values else None

    def best_epoch(
        self,
        metric: str = "val_loss",
        direction: Literal["min", "max"] = "min",
    ) -> int:
        """
        Return the epoch index at which ``metric`` was best.

        Args:
            metric (str): Name of the metric to inspect (e.g. ``"val_loss"``).
                Defaults to ``"val_loss"``.
            direction (Literal["min", "max"]): Whether a lower (``"min"``) or
                higher (``"max"``) value is considered better. Defaults to
                ``"min"``.

        Returns:
            int: Epoch index at which ``metric`` achieved its best value.

        Raises:
            ValueError: If ``metric`` is not found in the recorded metrics.

        """
        available = self.metric_names()
        if metric not in available:
            msg = (
                f"Metric '{metric}' not found. "
                f"Available metrics: {available}."
            )
            raise ValueError(msg)

        entries = self.metrics().where(name=metric).values()
        best_entry = min(entries, key=lambda e: e.value) if direction == "min" else max(entries, key=lambda e: e.value)
        return best_entry.epoch_idx

    @property
    def last_epoch(self) -> int | None:
        """
        The epoch the model is currently at after training.

        If an :class:`EarlyStopping` callback with ``restore_best=True`` ran
        and successfully restored model weights, this returns the restored
        (best) epoch. Otherwise, returns the last recorded epoch index.
        Returns ``None`` if no epochs were recorded.

        Returns:
            int | None: Current model-state epoch, or ``None``.

        """
        es = self.early_stopping_result
        if es is not None and es.restored and es.best_epoch is not None:
            return es.best_epoch
        if self.epoch_indices:
            return self.epoch_indices[-1]
        return None
