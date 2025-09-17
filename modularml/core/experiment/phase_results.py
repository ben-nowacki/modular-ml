from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from matplotlib import pyplot as plt

from modularml.visualization.common.parity_plot import plot_parity_from_node_outputs

if TYPE_CHECKING:
    import pandas as pd

    from modularml.core.data_structures.node_outputs import NodeOutputs
    from modularml.core.experiment.experiment import Experiment
    from modularml.core.loss.loss_collection import LossCollection


@dataclass
class TrainingResult:
    """
    Container for results of a completed TrainingPhase.

    Attributes:
        label (str):
            Label of the training phase.
        final_train_loss (LossCollection):
            Training losses averaged over the final epoch.
        train_losses (list[LossCollection]):
            Per-epoch training losses.
        final_val_loss (LossCollection | None):
            Validation losses averaged over the final epoch, if validation was performed.
        val_losses (list[LossCollection] | None):
            Per-epoch validation losses, if validation was performed.
        train_outputs (NodeOutputs):
            Model outputs from the final training epoch.
        last_epoch (int):
            Index of the last completed epoch (0-based).
        stopped_early (bool):
            Whether training ended early due to early stopping.

    """

    phase_label: str
    final_train_loss: LossCollection  # last epoch
    train_losses: list[LossCollection]  # per-epoch

    final_val_loss: LossCollection | None  # last epoch, optional
    val_losses: list[LossCollection] | None  # per-epoch, optional

    train_outputs: NodeOutputs  # final epoch outputs
    last_epoch: int
    stopped_early: bool = False

    def plot_losses(self, *, show_aux: bool = False, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot training (and validation, if available) loss vs. epoch.

        Args:
            show_aux (bool): If True, also plot trainable and auxiliary loss separately.
            ax (matplotlib.Axes | None): Optional axis to plot on. If None, a new figure is created.

        Returns:
            matplotlib.Figure, matplotlib.Axes

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))
        else:
            fig = ax.figure

        # Training losses
        t_tot = [lc.total for lc in self.train_losses]
        ax.plot(t_tot, ".-", label="Total Train Loss")

        if show_aux:
            ax.plot([lc.trainable for lc in self.train_losses], ".-", label="Trainable Train Loss")
            ax.plot([lc.auxiliary for lc in self.train_losses], ".-", label="Auxiliary Train Loss")

        # Validation losses
        if self.val_losses is not None and len(self.val_losses) > 0:
            v_tot = [lc.total for lc in self.val_losses]
            ax.plot(v_tot, ".--", label="Total Val Loss")

            if show_aux:
                ax.plot([lc.trainable for lc in self.val_losses], ".--", label="Trainable Val Loss")
                ax.plot([lc.auxiliary for lc in self.val_losses], ".--", label="Auxiliary Val Loss")

        ax.set_title(f"Loss vs Epoch ({self.phase_label})", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        return fig, ax

    def plot_parity(
        self,
        node: str,
        *,
        ax: plt.Axes | None = None,
        plot_style: str = "scatter",
        colors=None,
        cmap=plt.cm.Blues,
        gridsize: int = 25,
        alpha: float = 0.5,
        s: int = 12,
        line_style: str = ":",
        line_width: float = 1.5,
        line_color: str = "black",
        equal_aspect: bool = True,
        auto_bounds: bool = True,
        margin_frac: float = 0.05,
        dim: int | None = None,
        flatten: bool = False,
        title: str | None = None,
        kde_bw: float | None = None,
        kde_levels: int = 10,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        return plot_parity_from_node_outputs(
            outputs=self.train_outputs,
            node=node,
            title=title or f"Training {self.label}: {node}",
            ax=ax,
            plot_style=plot_style,
            colors=colors,
            cmap=cmap,
            gridsize=gridsize,
            alpha=alpha,
            s=s,
            line_style=line_style,
            line_width=line_width,
            line_color=line_color,
            equal_aspect=equal_aspect,
            auto_bounds=auto_bounds,
            margin_frac=margin_frac,
            dim=dim,
            flatten=flatten,
            kde_bw=kde_bw,
            kde_levels=kde_levels,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
        )


@dataclass
class EvaluationResult:
    """
    Container for results of a completed EvaluationPhase.

    Attributes:
        label (str):
            Label of the evaluation phase.
        losses (LossCollection | None):
            Losses computed during evaluation, if any losses were specified.
        outputs (NodeOutputs):
            Aggregated outputs from all nodes across evaluation batches.

    """

    phase_label: str
    losses: LossCollection | None
    outputs: NodeOutputs

    def plot_parity(
        self,
        node: str,
        *,
        ax: plt.Axes | None = None,
        plot_style: str = "scatter",
        colors=None,
        cmap=plt.cm.Blues,
        gridsize: int = 25,
        alpha: float = 0.5,
        s: int = 12,
        line_style: str = ":",
        line_width: float = 1.5,
        line_color: str = "black",
        equal_aspect: bool = True,
        auto_bounds: bool = True,
        margin_frac: float = 0.05,
        dim: int | None = None,
        flatten: bool = False,
        title: str | None = None,
        kde_bw: float | None = None,
        kde_levels: int = 10,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        return plot_parity_from_node_outputs(
            outputs=self.outputs,
            node=node,
            title=title or f"Training {self.phase_label}: {node}",
            ax=ax,
            plot_style=plot_style,
            colors=colors,
            cmap=cmap,
            gridsize=gridsize,
            alpha=alpha,
            s=s,
            line_style=line_style,
            line_width=line_width,
            line_color=line_color,
            equal_aspect=equal_aspect,
            auto_bounds=auto_bounds,
            margin_frac=margin_frac,
            dim=dim,
            flatten=flatten,
            kde_bw=kde_bw,
            kde_levels=kde_levels,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
        )

    def get_unscaled_outputs(
        self,
        node: str,
        experiment: Experiment,
        *,
        component: Literal["features", "targets"] = "targets",
        which: Literal["all", "last"] = "all",
        strict: bool = True,
    ) -> pd.DataFrame:
        return experiment.inverse_transform_node_outputs(
            all_outputs=self.outputs,
            node=node,
            component=component,
            which=which,
            strict=strict,
        )
