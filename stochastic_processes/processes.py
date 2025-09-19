"""Definitions of stochastic processes supported by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """Container holding the outcome of a stochastic process simulation."""

    name: str
    times: np.ndarray
    paths: np.ndarray
    description: str
    parameters: Mapping[str, Any]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def parameter_summary(self) -> str:
        """Return a human readable summary of the simulation parameters."""

        if not self.parameters:
            return ""
        ordered = ", ".join(f"{key}={value}" for key, value in self.parameters.items())
        return ordered


ParameterType = Callable[[str], Any]


@dataclass(frozen=True)
class ProcessParameter:
    """Description of a single argument exposed by a stochastic process."""

    name: str
    type: ParameterType
    default: Any
    help: str
    choices: Optional[Sequence[Any]] = None
    validator: Optional[Callable[[Any], None]] = None


@dataclass(frozen=True)
class ProcessDefinition:
    """Metadata and implementation details for an available process."""

    key: str
    title: str
    description: str
    category: str
    parameters: Sequence[ProcessParameter]
    simulator: Callable[..., SimulationResult]

    def simulate(self, *, n_paths: int, seed: Optional[int], **kwargs: Any) -> SimulationResult:
        """Execute the simulator associated with this definition."""

        return self.simulator(n_paths=n_paths, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Process implementations
# ---------------------------------------------------------------------------

def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value}.")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, received {value}.")


def _validate_probability(value: float, name: str = "p") -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, received {value}.")


def _validate_num_paths(n_paths: int) -> None:
    if n_paths < 1:
        raise ValueError("At least one sample path must be generated.")


def _prepare_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_walk(
    *,
    steps: int = 250,
    p: float = 0.5,
    step_size: float = 1.0,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_probability(p, "p")
    _validate_positive(step_size, "step_size")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    increments = rng.choice(
        np.array([step_size, -step_size]),
        size=(n_paths, steps),
        p=[p, 1 - p],
    )
    cumulative = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    times = np.arange(steps + 1)
    metadata = {
        "plot_style": "step",
        "x_label": "Step",
        "y_label": "Position",
    }
    return SimulationResult(
        name="Simple Random Walk",
        times=times,
        paths=cumulative,
        description=(
            "Discrete-time process that adds ±step_size increments with "
            f"probability p={p}."
        ),
        parameters={"steps": steps, "p": p, "step_size": step_size},
        metadata=metadata,
    )


def _autoregressive(
    *,
    steps: int = 250,
    phi: float = 0.8,
    sigma: float = 1.0,
    x0: float = 0.0,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_non_negative(sigma, "sigma")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    epsilon = rng.normal(loc=0.0, scale=sigma, size=(n_paths, steps))
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = x0
    for t in range(steps):
        paths[:, t + 1] = phi * paths[:, t] + epsilon[:, t]
    times = np.arange(steps + 1)
    metadata = {"plot_style": "line", "x_label": "Step", "y_label": "Value"}
    return SimulationResult(
        name="Autoregressive AR(1)",
        times=times,
        paths=paths,
        description=(
            "Discrete-time process X_t = phi * X_{t-1} + epsilon_t where the "
            "innovations are Gaussian."
        ),
        parameters={"steps": steps, "phi": phi, "sigma": sigma, "x0": x0},
        metadata=metadata,
    )


def _brownian_motion(
    *,
    t_max: float = 1.0,
    steps: int = 500,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    _validate_positive(t_max, "t_max")
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_non_negative(sigma, "sigma")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    dt = t_max / steps
    increments = rng.normal(
        loc=mu * dt,
        scale=sigma * np.sqrt(dt),
        size=(n_paths, steps),
    )
    paths = np.concatenate(
        [np.full((n_paths, 1), x0), x0 + np.cumsum(increments, axis=1)], axis=1
    )
    times = np.linspace(0.0, t_max, steps + 1)
    metadata = {
        "plot_style": "line",
        "x_label": "Time",
        "y_label": "Value",
    }
    return SimulationResult(
        name="Brownian Motion",
        times=times,
        paths=paths,
        description=(
            "Continuous-time process with normally distributed increments and "
            "variance growing linearly with time."
        ),
        parameters={
            "t_max": t_max,
            "steps": steps,
            "mu": mu,
            "sigma": sigma,
            "x0": x0,
        },
        metadata=metadata,
    )


def _geometric_brownian(
    *,
    t_max: float = 1.0,
    steps: int = 500,
    mu: float = 0.1,
    sigma: float = 0.2,
    s0: float = 1.0,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    _validate_positive(t_max, "t_max")
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_non_negative(sigma, "sigma")
    _validate_positive(s0, "s0")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    dt = t_max / steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    shocks = rng.normal(loc=0.0, scale=1.0, size=(n_paths, steps))
    log_returns = drift + diffusion * shocks
    cumulative_log_returns = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = s0 * np.exp(cumulative_log_returns)
    times = np.linspace(0.0, t_max, steps + 1)
    metadata = {
        "plot_style": "line",
        "x_label": "Time",
        "y_label": "Value",
    }
    return SimulationResult(
        name="Geometric Brownian Motion",
        times=times,
        paths=paths,
        description=(
            "Continuous-time process commonly used for modelling asset prices."
        ),
        parameters={
            "t_max": t_max,
            "steps": steps,
            "mu": mu,
            "sigma": sigma,
            "s0": s0,
        },
        metadata=metadata,
    )


def _poisson_process(
    *,
    rate: float = 5.0,
    t_max: float = 10.0,
    steps: int = 500,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    _validate_positive(rate, "rate")
    _validate_positive(t_max, "t_max")
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    dt = t_max / steps
    increments = rng.poisson(lam=rate * dt, size=(n_paths, steps))
    paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    times = np.linspace(0.0, t_max, steps + 1)
    metadata = {
        "plot_style": "step",
        "x_label": "Time",
        "y_label": "Event count",
    }
    return SimulationResult(
        name="Poisson Process",
        times=times,
        paths=paths,
        description=(
            "Counting process with exponentially distributed inter-arrival times."
        ),
        parameters={"rate": rate, "t_max": t_max, "steps": steps},
        metadata=metadata,
    )


def _ornstein_uhlenbeck(
    *,
    t_max: float = 5.0,
    steps: int = 500,
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 0.3,
    x0: float = 0.0,
    n_paths: int,
    seed: Optional[int],
) -> SimulationResult:
    _validate_positive(t_max, "t_max")
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    _validate_positive(theta, "theta")
    _validate_non_negative(sigma, "sigma")
    _validate_num_paths(n_paths)

    rng = _prepare_rng(seed)
    dt = t_max / steps
    shocks = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, steps))
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = x0
    for t in range(steps):
        paths[:, t + 1] = (
            paths[:, t]
            + theta * (mu - paths[:, t]) * dt
            + sigma * shocks[:, t]
        )
    times = np.linspace(0.0, t_max, steps + 1)
    metadata = {
        "plot_style": "line",
        "x_label": "Time",
        "y_label": "Value",
    }
    return SimulationResult(
        name="Ornstein-Uhlenbeck Process",
        times=times,
        paths=paths,
        description="Mean-reverting continuous-time Gaussian process.",
        parameters={
            "t_max": t_max,
            "steps": steps,
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "x0": x0,
        },
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

AVAILABLE_PROCESSES: Dict[str, ProcessDefinition] = {
    "random_walk": ProcessDefinition(
        key="random_walk",
        title="Simple Random Walk",
        description="Classic ±1 walk with configurable probability and step size.",
        category="discrete-time",
        parameters=[
            ProcessParameter(
                name="steps",
                type=int,
                default=250,
                help="Number of steps to simulate.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
            ProcessParameter(
                name="p",
                type=float,
                default=0.5,
                help="Probability of taking a positive step.",
                validator=lambda value: _validate_probability(value, "p"),
            ),
            ProcessParameter(
                name="step_size",
                type=float,
                default=1.0,
                help="Magnitude of each step.",
                validator=lambda value: _validate_positive(value, "step_size"),
            ),
        ],
        simulator=_random_walk,
    ),
    "ar1": ProcessDefinition(
        key="ar1",
        title="Autoregressive AR(1)",
        description="Discrete-time Gaussian AR(1) process.",
        category="discrete-time",
        parameters=[
            ProcessParameter(
                name="steps",
                type=int,
                default=250,
                help="Number of observations to generate.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
            ProcessParameter(
                name="phi",
                type=float,
                default=0.8,
                help="Autoregressive coefficient.",
            ),
            ProcessParameter(
                name="sigma",
                type=float,
                default=1.0,
                help="Standard deviation of the innovation term.",
                validator=lambda value: _validate_non_negative(value, "sigma"),
            ),
            ProcessParameter(
                name="x0",
                type=float,
                default=0.0,
                help="Initial value of the process.",
            ),
        ],
        simulator=_autoregressive,
    ),
    "brownian_motion": ProcessDefinition(
        key="brownian_motion",
        title="Brownian Motion",
        description="Continuous-time Brownian motion with drift.",
        category="continuous-time",
        parameters=[
            ProcessParameter(
                name="t_max",
                type=float,
                default=1.0,
                help="Time horizon for the simulation.",
                validator=lambda value: _validate_positive(value, "t_max"),
            ),
            ProcessParameter(
                name="steps",
                type=int,
                default=500,
                help="Number of discretisation steps.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
            ProcessParameter(
                name="mu",
                type=float,
                default=0.0,
                help="Drift term of the process.",
            ),
            ProcessParameter(
                name="sigma",
                type=float,
                default=1.0,
                help="Volatility parameter.",
                validator=lambda value: _validate_non_negative(value, "sigma"),
            ),
            ProcessParameter(
                name="x0",
                type=float,
                default=0.0,
                help="Initial value of the process.",
            ),
        ],
        simulator=_brownian_motion,
    ),
    "geometric_brownian_motion": ProcessDefinition(
        key="geometric_brownian_motion",
        title="Geometric Brownian Motion",
        description="Exponentially scaled Brownian motion used in finance.",
        category="continuous-time",
        parameters=[
            ProcessParameter(
                name="t_max",
                type=float,
                default=1.0,
                help="Time horizon for the simulation.",
                validator=lambda value: _validate_positive(value, "t_max"),
            ),
            ProcessParameter(
                name="steps",
                type=int,
                default=500,
                help="Number of discretisation steps.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
            ProcessParameter(
                name="mu",
                type=float,
                default=0.1,
                help="Expected return / drift term.",
            ),
            ProcessParameter(
                name="sigma",
                type=float,
                default=0.2,
                help="Volatility parameter.",
                validator=lambda value: _validate_non_negative(value, "sigma"),
            ),
            ProcessParameter(
                name="s0",
                type=float,
                default=1.0,
                help="Initial asset level.",
                validator=lambda value: _validate_positive(value, "s0"),
            ),
        ],
        simulator=_geometric_brownian,
    ),
    "poisson_process": ProcessDefinition(
        key="poisson_process",
        title="Poisson Process",
        description="Counting process with a constant intensity.",
        category="continuous-time",
        parameters=[
            ProcessParameter(
                name="rate",
                type=float,
                default=5.0,
                help="Intensity (events per unit time).",
                validator=lambda value: _validate_positive(value, "rate"),
            ),
            ProcessParameter(
                name="t_max",
                type=float,
                default=10.0,
                help="Time horizon for the simulation.",
                validator=lambda value: _validate_positive(value, "t_max"),
            ),
            ProcessParameter(
                name="steps",
                type=int,
                default=500,
                help="Number of discretisation steps.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
        ],
        simulator=_poisson_process,
    ),
    "ornstein_uhlenbeck": ProcessDefinition(
        key="ornstein_uhlenbeck",
        title="Ornstein-Uhlenbeck",
        description="Mean-reverting Gaussian diffusion.",
        category="continuous-time",
        parameters=[
            ProcessParameter(
                name="t_max",
                type=float,
                default=5.0,
                help="Time horizon for the simulation.",
                validator=lambda value: _validate_positive(value, "t_max"),
            ),
            ProcessParameter(
                name="steps",
                type=int,
                default=500,
                help="Number of discretisation steps.",
                validator=lambda value: _validate_positive(value, "steps"),
            ),
            ProcessParameter(
                name="theta",
                type=float,
                default=1.0,
                help="Speed of mean reversion.",
                validator=lambda value: _validate_positive(value, "theta"),
            ),
            ProcessParameter(
                name="mu",
                type=float,
                default=0.0,
                help="Long-run mean of the process.",
            ),
            ProcessParameter(
                name="sigma",
                type=float,
                default=0.3,
                help="Volatility parameter.",
                validator=lambda value: _validate_non_negative(value, "sigma"),
            ),
            ProcessParameter(
                name="x0",
                type=float,
                default=0.0,
                help="Initial value of the process.",
            ),
        ],
        simulator=_ornstein_uhlenbeck,
    ),
}

