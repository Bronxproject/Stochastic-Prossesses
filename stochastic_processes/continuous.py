"""Continuous time stochastic process simulators."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import SimulationResult


def _time_grid(t_max: float, dt: float) -> np.ndarray:
    steps = int(np.ceil(t_max / dt))
    return np.linspace(0.0, steps * dt, steps + 1)


def simulate_poisson_process(params: Dict[str, float], seed: Optional[int] = None) -> SimulationResult:
    """Simulate a Poisson counting process."""

    rate = float(params["rate"])
    t_max = float(params["t_max"])
    dt = float(params["dt"])
    paths = int(params["paths"])

    time = _time_grid(t_max, dt)
    num_steps = time.size - 1
    rng = np.random.default_rng(seed)
    increments = rng.poisson(rate * dt, size=(paths, num_steps))
    counts = np.concatenate([
        np.zeros((paths, 1), dtype=float),
        np.cumsum(increments, axis=1),
    ], axis=1)
    metadata = {
        "rate": rate,
        "t_max": float(time[-1]),
        "dt": dt,
        "paths": paths,
        "process": "poisson_process",
    }
    return SimulationResult(time=time, values=counts, metadata=metadata)


def simulate_brownian_motion(params: Dict[str, float], seed: Optional[int] = None) -> SimulationResult:
    """Simulate a Brownian motion with optional drift and volatility."""

    drift = float(params["drift"])
    volatility = float(params["volatility"])
    t_max = float(params["t_max"])
    dt = float(params["dt"])
    paths = int(params["paths"])

    time = _time_grid(t_max, dt)
    num_steps = time.size - 1
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(paths, num_steps))
    paths_values = np.concatenate([
        np.zeros((paths, 1), dtype=float),
        np.cumsum(increments, axis=1),
    ], axis=1)
    values = drift * time + volatility * paths_values
    metadata = {
        "drift": drift,
        "volatility": volatility,
        "t_max": float(time[-1]),
        "dt": dt,
        "paths": paths,
        "process": "brownian_motion",
    }
    return SimulationResult(time=time, values=values, metadata=metadata)


def simulate_geometric_brownian_motion(
    params: Dict[str, float], seed: Optional[int] = None
) -> SimulationResult:
    """Simulate a geometric Brownian motion (commonly used in finance)."""

    mu = float(params["mu"])
    sigma = float(params["sigma"])
    initial = float(params["initial"])
    t_max = float(params["t_max"])
    dt = float(params["dt"])
    paths = int(params["paths"])

    time = _time_grid(t_max, dt)
    num_steps = time.size - 1
    rng = np.random.default_rng(seed)
    normal_increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(paths, num_steps))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * normal_increments
    log_paths = np.concatenate([
        np.zeros((paths, 1), dtype=float),
        np.cumsum(increments, axis=1),
    ], axis=1)
    values = initial * np.exp(log_paths)
    metadata = {
        "mu": mu,
        "sigma": sigma,
        "initial": initial,
        "t_max": float(time[-1]),
        "dt": dt,
        "paths": paths,
        "process": "geometric_brownian_motion",
    }
    return SimulationResult(time=time, values=values, metadata=metadata)


def simulate_ornstein_uhlenbeck(params: Dict[str, float], seed: Optional[int] = None) -> SimulationResult:
    """Simulate an Ornstein-Uhlenbeck mean-reverting process."""

    theta = float(params["theta"])
    mu = float(params["mu"])
    sigma = float(params["sigma"])
    initial = float(params["initial"])
    t_max = float(params["t_max"])
    dt = float(params["dt"])
    paths = int(params["paths"])

    time = _time_grid(t_max, dt)
    num_steps = time.size - 1
    rng = np.random.default_rng(seed)
    normals = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(paths, num_steps))
    values = np.empty((paths, num_steps + 1), dtype=float)
    values[:, 0] = initial
    for t in range(1, num_steps + 1):
        values[:, t] = (
            values[:, t - 1]
            + theta * (mu - values[:, t - 1]) * dt
            + sigma * normals[:, t - 1]
        )
    metadata = {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "initial": initial,
        "t_max": float(time[-1]),
        "dt": dt,
        "paths": paths,
        "process": "ornstein_uhlenbeck",
    }
    return SimulationResult(time=time, values=values, metadata=metadata)
