"""Plotting utilities for stochastic process simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .base import ProcessSpec, SimulationResult


def plot_simulation(
    result: SimulationResult,
    spec: ProcessSpec,
    *,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot the simulated paths.

    Parameters
    ----------
    result:
        Simulation outcome containing the time grid and sample paths.
    spec:
        Metadata for the process used to annotate the plot.
    output_path:
        If provided, save the figure to this location.
    show:
        Display the figure interactively when ``True`` and ``output_path`` is
        not provided. ``show`` is ignored when ``output_path`` is specified.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, path in enumerate(result.values):
        label = "Sample path" if idx == 0 else None
        ax.plot(result.time, path, alpha=0.75, label=label)

    ax.set_title(spec.name)
    ax.set_xlabel(spec.time_label)
    ax.set_ylabel(spec.value_label)
    ax.grid(True, linestyle="--", alpha=0.4)

    text_lines = [f"Category: {spec.category.capitalize()}"]
    for key, value in result.metadata.items():
        if key == "process":
            continue
        text_lines.append(f"{key}: {value}")
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    elif show:
        plt.show()
    plt.close(fig)
