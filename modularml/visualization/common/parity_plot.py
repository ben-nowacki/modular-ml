import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from modularml.core.data_structures.node_outputs import NodeOutputs


def parity_plot(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    plot_style: str = "hexbin",
    colors=None,
    cmap=plt.cm.Blues,
    gridsize: int = 25,
    alpha: float = 0.5,
    s: int = 12,
    line_style: str = ":",
    line_width: float = 1.5,
    line_color: str = "black",
    equal_aspect: bool = True,
    kde_bw: float | None = None,
    kde_levels: int = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a parity plot comparing predictions to true values.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground-truth values.
        ax (plt.Axes | None): Axis to plot on. If None, creates a new figure/axis.
        plot_style (str): Plot type. Options:
            - "scatter": raw scatter of points.
            - "hexbin": hexbin density plot (default).
            - "kde": 2D kernel density estimate with contour lines.
        colors (array-like | None): Colors for scatter points (scatter mode).
        cmap (Colormap): Colormap for density modes.
        gridsize (int): Grid resolution for hexbin.
        alpha (float): Transparency for scatter or KDE shading.
        s (int): Marker size for scatter.
        line_style (str): Style of 1:1 reference line.
        line_width (float): Width of 1:1 reference line.
        line_color (str): Color of 1:1 reference line.
        equal_aspect (bool): If True, enforce equal aspect ratio.
        kde_bw (float | None): Bandwidth override for KDE. Default = auto.
        kde_levels (int): Number of contour levels for KDE plot.
        vmin (float): Minimum value for color map normalization.
        vmax (float): Maximum value for color map normalization.
        colorbar: whether to plot a colorbar.

    Returns:
        (matplotlib.Figure, matplotlib.Axes)

    Notes:
        - Use "hexbin" or "kde" for large datasets for readability.
        - KDE can be slow on very large arrays.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if plot_style == "scatter":
        ax.scatter(true, pred, c=colors, alpha=alpha, s=s)

    elif plot_style == "hexbin":
        hb = ax.hexbin(true, pred, gridsize=gridsize, bins="log", cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            fig.colorbar(hb, ax=ax, label="log10(N)")

    elif plot_style == "kde":
        values = np.vstack([true, pred])
        kde = gaussian_kde(values, bw_method=kde_bw)
        xmin, xmax = true.min(), true.max()
        ymin, ymax = pred.min(), pred.max()
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        low = zz.min() if vmin is None else vmin
        high = zz.max() if vmax is None else vmax
        levels = np.linspace(low, high, kde_levels)
        mod_cmap = cmap.copy()
        mod_cmap.set_under("white")  # Force base level to be white
        cf = ax.contourf(
            xx,
            yy,
            zz,
            levels=levels,
            cmap=mod_cmap,
            alpha=alpha,
            extend="min",
        )
        if colorbar:
            fig.colorbar(cf, ax=ax, label="density")

    else:
        msg = f"Unsupported plot_style: {plot_style}"
        raise ValueError(msg)

    # 1:1 reference line
    ax.axline(
        (0, 0),
        (1, 1),
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
    )

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    return fig, ax


def plot_parity_from_node_outputs(
    outputs: NodeOutputs | pd.DataFrame,
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
    """
    Generate a parity plot from NodeOutputs.

    Args:
        outputs (NodeOutputs | pd.DataFrame): TrainingResult/EvaluationResult outputs \
            in raw form or as a Pandas DataFrame.
        node (str): Node label to plot.
        ax (plt.Axes | None): Axis to plot on. If None, creates a new figure/axis.
        plot_style (str): One of {"scatter", "hexbin", "kde"}.
        colors (array-like | None): Colors for scatter points (scatter mode).
        cmap (Colormap): Colormap for density modes.
        gridsize (int): Grid resolution for hexbin.
        alpha (float): Transparency for scatter or KDE shading.
        s (int): Marker size for scatter.
        line_style (str): Style of 1:1 reference line.
        line_width (float): Width of 1:1 reference line.
        line_color (str): Color of 1:1 reference line.
        equal_aspect (bool): If True, enforce equal aspect ratio.
        auto_bounds (bool): Auto-adjust plot limits. Defaults to True.
        margin_frac (float): Percent of axis range to add as empty spacing. Defaults to 0.05.
        dim (int | None): Select dimension if multi-D.
        flatten (bool): Flatten multi-D outputs to 1D.
        title (str | None): Optional title.
        kde_bw (float | None): Bandwidth override for KDE. Default = auto.
        kde_levels (int): Number of contour levels for KDE plot.
        vmin (float): Minimum value for color map normalization.
        vmax (float): Maximum value for color map normalization.
        colorbar (bool): Whether to add a colorbar. Defaulst to True.

    Returns:
        (matplotlib.Figure, matplotlib.Axes)

    """
    df = outputs
    if isinstance(outputs, NodeOutputs):
        df = outputs.to_dataframe()

    if node not in df["node"].unique():
        msg = f"Node `{node}` not found. Available: {df['node'].unique()}"
        raise ValueError(msg)

    df_filt = df[df["node"] == node]
    y_pred = np.vstack(df_filt["output"].to_numpy())
    y_true = np.vstack(df_filt["target"].to_numpy())

    # Handle dimensionality
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        if flatten:
            y_pred = y_pred.ravel()
            y_true = y_true.ravel()
        elif dim is not None:
            y_pred = y_pred[:, dim]
            y_true = y_true[:, dim]
        else:
            msg = (
                f"Node `{node}` outputs/targets are multi-dimensional: {y_pred.shape}. Specify `dim` or `flatten=True`."
            )
            raise ValueError(msg)
    else:
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

    fig, ax = parity_plot(
        pred=y_pred,
        true=y_true,
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
        kde_bw=kde_bw,
        kde_levels=kde_levels,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
    )

    if auto_bounds:
        all_vals = np.concatenate([y_pred, y_true])
        lo, hi = all_vals.min(), all_vals.max()
        delta = (hi - lo) * margin_frac
        ax.set_xlim(lo - delta, hi + delta)
        ax.set_ylim(lo - delta, hi + delta)

    ax.set_title(title or f"Parity Plot: {node}", fontsize=10)
    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)

    return fig, ax
