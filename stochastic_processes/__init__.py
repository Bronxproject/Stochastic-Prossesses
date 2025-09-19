"""Utilities for simulating and visualising stochastic processes.

This package exposes the available process registry and the plotting helper
that backs the command line interface defined in :mod:`simulate`.
"""

from .processes import AVAILABLE_PROCESSES, ProcessDefinition, SimulationResult
from .plotting import plot_simulation

__all__ = [
    "AVAILABLE_PROCESSES",
    "ProcessDefinition",
    "SimulationResult",
    "plot_simulation",
]
