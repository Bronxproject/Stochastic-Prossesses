"""Base data structures used by the stochastic process simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SimulationResult:
    """Holds the outcome of a stochastic process simulation.

    Attributes
    ----------
    time:
        One dimensional array describing the time or step index.
    values:
        Two dimensional array with shape ``(paths, len(time))`` describing the
        simulated sample paths.
    metadata:
        Arbitrary metadata about the simulation (for example parameter values
        used to generate the paths).
    """

    time: np.ndarray
    values: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time.ndim != 1:
            raise ValueError("time must be a 1-D array")
        if self.values.ndim != 2:
            raise ValueError("values must be a 2-D array")
        if self.values.shape[1] != self.time.shape[0]:
            raise ValueError(
                "values second dimension must equal the length of the time axis"
            )


@dataclass
class ParameterSpec:
    """Specification of a configurable parameter for a stochastic process."""

    name: str
    type: Callable[[str], Any]
    default: Any
    help: str
    choices: Optional[Sequence[Any]] = None
    validator: Optional[Callable[[Any], None]] = None

    def parse(self, raw: str) -> Any:
        """Parse a string representation into the correct data type.

        Parameters
        ----------
        raw:
            String provided by the user.
        """

        value = self.type(raw)
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Invalid value '{value}' for parameter '{self.name}'. "
                f"Allowed values are: {self.choices}."
            )
        if self.validator is not None:
            self.validator(value)
        return value


@dataclass
class ProcessSpec:
    """Metadata and callable for a stochastic process simulation."""

    key: str
    name: str
    category: str
    description: str
    simulator: Callable[[Mapping[str, Any], Optional[int]], SimulationResult]
    parameters: Mapping[str, ParameterSpec]
    time_label: str = "Time"
    value_label: str = "Value"

    def coerce_parameters(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Combine default values with optional overrides."""

        values: Dict[str, Any] = {name: spec.default for name, spec in self.parameters.items()}
        if overrides:
            for name, value in overrides.items():
                if name not in self.parameters:
                    raise KeyError(f"Unknown parameter '{name}' for process '{self.key}'.")
                spec = self.parameters[name]
                if isinstance(value, str):
                    value = spec.parse(value)
                else:
                    if spec.choices is not None and value not in spec.choices:
                        raise ValueError(
                            f"Invalid value '{value}' for parameter '{name}'. "
                            f"Allowed values are: {spec.choices}."
                        )
                    if spec.validator is not None:
                        spec.validator(value)
                values[name] = value
        return values


def positive_int(value: int) -> None:
    if value <= 0:
        raise ValueError("value must be a positive integer")


def positive_float(value: float) -> None:
    if value <= 0:
        raise ValueError("value must be a positive number")


def probability(value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError("probability values must be within [0, 1]")
