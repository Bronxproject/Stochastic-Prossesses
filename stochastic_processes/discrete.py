"""Discrete time stochastic process simulators."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import SimulationResult


def simulate_simple_random_walk(params: Dict[str, float], seed: Optional[int] = None) -> SimulationResult:
    """Simulate a simple symmetric (or biased) random walk."""

    steps = int(params["steps"])
    start = float(params["start"])
    p = float(params["p"])
    paths = int(params["paths"])
    step_size = float(params["step_size"])

    rng = np.random.default_rng(seed)
    increments = rng.choice(
        [-step_size, step_size], size=(paths, steps), p=[1 - p, p]
    )
    start_column = np.full((paths, 1), start, dtype=float)
    values = np.concatenate(
        [start_column, start + np.cumsum(increments, axis=1)], axis=1
    )
    time = np.arange(steps + 1, dtype=int)
    metadata = {
        "steps": steps,
        "start": start,
        "p": p,
        "paths": paths,
        "step_size": step_size,
        "process": "simple_random_walk",
    }
    return SimulationResult(time=time, values=values, metadata=metadata)


def simulate_autoregressive(params: Dict[str, float], seed: Optional[int] = None) -> SimulationResult:
    """Simulate an autoregressive AR(1) process."""

    steps = int(params["steps"])
    phi = float(params["phi"])
    sigma = float(params["sigma"])
    mean = float(params["mean"])
    start = float(params["start"])
    paths = int(params["paths"])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=0.0, scale=sigma, size=(paths, steps))
    values = np.empty((paths, steps + 1), dtype=float)
    values[:, 0] = start
    for t in range(1, steps + 1):
        values[:, t] = mean + phi * (values[:, t - 1] - mean) + shocks[:, t - 1]
    time = np.arange(steps + 1, dtype=int)
    metadata = {
        "steps": steps,
        "phi": phi,
        "sigma": sigma,
        "mean": mean,
        "start": start,
        "paths": paths,
        "process": "autoregressive",
    }
    return SimulationResult(time=time, values=values, metadata=metadata)


def simulate_two_state_markov_chain(
    params: Dict[str, float], seed: Optional[int] = None
) -> SimulationResult:
    """Simulate a two state discrete time Markov chain."""

    steps = int(params["steps"])
    p00 = float(params["p00"])
    p11 = float(params["p11"])
    start_state = int(params["start_state"])
    paths = int(params["paths"])

    if start_state not in (0, 1):
        raise ValueError("start_state must be 0 or 1 for a two-state Markov chain")

    transition = np.array([[p00, 1 - p00], [1 - p11, p11]], dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty((paths, steps + 1), dtype=int)
    values[:, 0] = start_state
    for t in range(1, steps + 1):
        prev_states = values[:, t - 1]
        random_values = rng.random(paths)
        next_states = np.where(
            random_values < transition[prev_states, 0], 0, 1
        )
        values[:, t] = next_states

    time = np.arange(steps + 1, dtype=int)
    metadata = {
        "steps": steps,
        "p00": p00,
        "p11": p11,
        "start_state": start_state,
        "paths": paths,
        "process": "two_state_markov_chain",
    }
    return SimulationResult(time=time, values=values.astype(float), metadata=metadata)
