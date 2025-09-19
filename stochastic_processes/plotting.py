"""Plotting helpers for stochastic process simulations."""

from __future__ import annotations

import sys
from typing import Optional

import matplotlib

from .processes import SimulationResult


def _load_pyplot(show: bool):
    """Load :mod:`matplotlib.pyplot` with a backend suitable for the environment."""

    if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt  # type: ignore

        return plt

    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt  # type: ignore

    return plt


def plot_simulation(
    result: SimulationResult,
    *,
    output: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Render the sample paths of a simulation result.

    Parameters
    ----------
    result:
        The outcome returned by the simulator.
    output:
        Optional path where the figure should be written.
    show:
        Whether to display the figure using :func:`matplotlib.pyplot.show`.

    Returns
    -------
    Optional[str]
        The path of the written figure when ``output`` is provided.
    """

    plt = _load_pyplot(show)
    fig, ax = plt.subplots(figsize=(10, 6))

    style = result.metadata.get("plot_style", "line")
    x_label = result.metadata.get("x_label", "Time")
    y_label = result.metadata.get("y_label", "Value")

    for index, path in enumerate(result.paths):
        label = None
        if result.paths.shape[0] > 1:
            label = f"Path {index + 1}"

        if style == "step":
            ax.step(result.times, path, where="post", alpha=0.85, linewidth=1.5, label=label)
        else:
            ax.plot(result.times, path, alpha=0.9, linewidth=1.5, label=label)

    ax.set_title(result.name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if result.paths.shape[0] > 1:
        ax.legend(loc="best", frameon=False)

    parameter_summary = result.parameter_summary()
    if parameter_summary:
        ax.text(
            0.5,
            -0.2,
            parameter_summary,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
        )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if output:
        fig.savefig(output, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output

