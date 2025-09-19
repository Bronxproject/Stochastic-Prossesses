"""Registry of available stochastic process simulators."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from .base import (
    ParameterSpec,
    ProcessSpec,
    SimulationResult,
    positive_float,
    positive_int,
    probability,
)
from . import continuous, discrete


PROCESS_REGISTRY: Dict[str, ProcessSpec] = {}


def _register(spec: ProcessSpec) -> None:
    PROCESS_REGISTRY[spec.key] = spec


_register(
    ProcessSpec(
        key="simple_random_walk",
        name="Simple Random Walk",
        category="discrete",
        description="Biased or unbiased discrete-time random walk with configurable step size.",
        simulator=discrete.simulate_simple_random_walk,
        parameters={
            "steps": ParameterSpec(
                name="steps",
                type=int,
                default=100,
                help="Number of steps to simulate.",
                validator=positive_int,
            ),
            "start": ParameterSpec(
                name="start",
                type=float,
                default=0.0,
                help="Starting value of the process.",
            ),
            "p": ParameterSpec(
                name="p",
                type=float,
                default=0.5,
                help="Probability of an upward move at each step.",
                validator=probability,
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
            "step_size": ParameterSpec(
                name="step_size",
                type=float,
                default=1.0,
                help="Magnitude of each step.",
                validator=positive_float,
            ),
        },
        time_label="Step",
        value_label="Position",
    )
)

_register(
    ProcessSpec(
        key="autoregressive",
        name="Autoregressive AR(1)",
        category="discrete",
        description="First order autoregressive process with Gaussian innovations.",
        simulator=discrete.simulate_autoregressive,
        parameters={
            "steps": ParameterSpec(
                name="steps",
                type=int,
                default=200,
                help="Number of steps to simulate.",
                validator=positive_int,
            ),
            "phi": ParameterSpec(
                name="phi",
                type=float,
                default=0.8,
                help="Autoregressive coefficient.",
            ),
            "sigma": ParameterSpec(
                name="sigma",
                type=float,
                default=0.5,
                help="Standard deviation of the innovations.",
                validator=positive_float,
            ),
            "mean": ParameterSpec(
                name="mean",
                type=float,
                default=0.0,
                help="Long-run mean of the process.",
            ),
            "start": ParameterSpec(
                name="start",
                type=float,
                default=0.0,
                help="Initial value of the process.",
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=3,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Step",
        value_label="Value",
    )
)

_register(
    ProcessSpec(
        key="two_state_markov_chain",
        name="Two-State Markov Chain",
        category="discrete",
        description="Discrete time Markov chain with two states and configurable transition probabilities.",
        simulator=discrete.simulate_two_state_markov_chain,
        parameters={
            "steps": ParameterSpec(
                name="steps",
                type=int,
                default=100,
                help="Number of steps to simulate.",
                validator=positive_int,
            ),
            "p00": ParameterSpec(
                name="p00",
                type=float,
                default=0.9,
                help="Probability of remaining in state 0.",
                validator=probability,
            ),
            "p11": ParameterSpec(
                name="p11",
                type=float,
                default=0.8,
                help="Probability of remaining in state 1.",
                validator=probability,
            ),
            "start_state": ParameterSpec(
                name="start_state",
                type=int,
                default=0,
                help="Initial state (0 or 1).",
                choices=[0, 1],
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Step",
        value_label="State",
    )
)

_register(
    ProcessSpec(
        key="poisson_process",
        name="Poisson Process",
        category="continuous",
        description="Counting process with exponentially distributed inter-arrival times.",
        simulator=continuous.simulate_poisson_process,
        parameters={
            "rate": ParameterSpec(
                name="rate",
                type=float,
                default=2.0,
                help="Average number of events per unit time.",
                validator=positive_float,
            ),
            "t_max": ParameterSpec(
                name="t_max",
                type=float,
                default=10.0,
                help="Total simulation time horizon.",
                validator=positive_float,
            ),
            "dt": ParameterSpec(
                name="dt",
                type=float,
                default=0.1,
                help="Time discretisation step used for sampling.",
                validator=positive_float,
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Time",
        value_label="Count",
    )
)

_register(
    ProcessSpec(
        key="brownian_motion",
        name="Brownian Motion",
        category="continuous",
        description="Brownian motion with drift and volatility parameters.",
        simulator=continuous.simulate_brownian_motion,
        parameters={
            "drift": ParameterSpec(
                name="drift",
                type=float,
                default=0.0,
                help="Drift component of the process.",
            ),
            "volatility": ParameterSpec(
                name="volatility",
                type=float,
                default=1.0,
                help="Volatility (diffusion coefficient).",
                validator=positive_float,
            ),
            "t_max": ParameterSpec(
                name="t_max",
                type=float,
                default=1.0,
                help="Total simulation time horizon.",
                validator=positive_float,
            ),
            "dt": ParameterSpec(
                name="dt",
                type=float,
                default=0.01,
                help="Time discretisation step used for sampling.",
                validator=positive_float,
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Time",
        value_label="Value",
    )
)

_register(
    ProcessSpec(
        key="geometric_brownian_motion",
        name="Geometric Brownian Motion",
        category="continuous",
        description="Geometric Brownian motion often used for modelling stock prices.",
        simulator=continuous.simulate_geometric_brownian_motion,
        parameters={
            "mu": ParameterSpec(
                name="mu",
                type=float,
                default=0.1,
                help="Drift of the log returns.",
            ),
            "sigma": ParameterSpec(
                name="sigma",
                type=float,
                default=0.2,
                help="Volatility of the process.",
                validator=positive_float,
            ),
            "initial": ParameterSpec(
                name="initial",
                type=float,
                default=1.0,
                help="Initial level of the process.",
                validator=positive_float,
            ),
            "t_max": ParameterSpec(
                name="t_max",
                type=float,
                default=1.0,
                help="Total simulation time horizon.",
                validator=positive_float,
            ),
            "dt": ParameterSpec(
                name="dt",
                type=float,
                default=0.01,
                help="Time discretisation step used for sampling.",
                validator=positive_float,
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Time",
        value_label="Value",
    )
)

_register(
    ProcessSpec(
        key="ornstein_uhlenbeck",
        name="Ornstein-Uhlenbeck",
        category="continuous",
        description="Mean-reverting diffusion process.",
        simulator=continuous.simulate_ornstein_uhlenbeck,
        parameters={
            "theta": ParameterSpec(
                name="theta",
                type=float,
                default=1.0,
                help="Speed of mean reversion.",
                validator=positive_float,
            ),
            "mu": ParameterSpec(
                name="mu",
                type=float,
                default=0.0,
                help="Long-run mean level.",
            ),
            "sigma": ParameterSpec(
                name="sigma",
                type=float,
                default=0.3,
                help="Volatility parameter.",
                validator=positive_float,
            ),
            "initial": ParameterSpec(
                name="initial",
                type=float,
                default=0.0,
                help="Initial level of the process.",
            ),
            "t_max": ParameterSpec(
                name="t_max",
                type=float,
                default=1.0,
                help="Total simulation time horizon.",
                validator=positive_float,
            ),
            "dt": ParameterSpec(
                name="dt",
                type=float,
                default=0.01,
                help="Time discretisation step used for sampling.",
                validator=positive_float,
            ),
            "paths": ParameterSpec(
                name="paths",
                type=int,
                default=5,
                help="Number of sample paths to simulate.",
                validator=positive_int,
            ),
        },
        time_label="Time",
        value_label="Value",
    )
)


def get_process_spec(key: str) -> ProcessSpec:
    return PROCESS_REGISTRY[key]


def list_processes() -> Iterable[ProcessSpec]:
    return PROCESS_REGISTRY.values()


def simulate_process(
    key: str, overrides: Optional[Mapping[str, object]] = None, seed: Optional[int] = None
) -> SimulationResult:
    spec = get_process_spec(key)
    params = spec.coerce_parameters(overrides)
    return spec.simulator(params, seed)
