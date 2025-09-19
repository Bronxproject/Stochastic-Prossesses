"""Command line entry point for the stochastic process simulator."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np

from stochastic_processes.processes import (
    AVAILABLE_PROCESSES,
    ProcessDefinition,
    ProcessParameter,
)
from stochastic_processes.plotting import plot_simulation


def _list_processes() -> None:
    """Print a summary of the processes available in the registry."""

    print("Available stochastic processes:\n")
    for key, definition in AVAILABLE_PROCESSES.items():
        print(f"- {key} ({definition.category})\n  {definition.title}\n  {definition.description}\n")


def _build_base_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Simulate sample paths for discrete and continuous stochastic processes.",
        formatter_class=formatter,
    )
    parser.add_argument(
        "process",
        nargs="?",
        choices=list(AVAILABLE_PROCESSES.keys()),
        help="Identifier of the process to simulate (use --list to see options).",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1,
        help="Number of independent sample paths to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the pseudo-random number generator.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination file for the generated figure (e.g. output.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the resulting plot in an interactive window.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available processes and exit.",
    )
    return parser


def _build_process_parser(definition: ProcessDefinition) -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description=f"{definition.title}\n\n{definition.description}",
        formatter_class=formatter,
    )
    for parameter in definition.parameters:
        argument_name = f"--{parameter.name.replace('_', '-')}"
        kwargs: Dict[str, object] = {
            "type": parameter.type,
            "default": parameter.default,
            "help": parameter.help,
        }
        if parameter.choices:
            kwargs["choices"] = parameter.choices
        parser.add_argument(argument_name, **kwargs)
    return parser


def _validate_paths(count: int, parser: argparse.ArgumentParser) -> None:
    if count < 1:
        parser.error("--paths must be at least 1.")


def _apply_validators(
    values: argparse.Namespace,
    parameters: List[ProcessParameter],
) -> None:
    for parameter in parameters:
        validator = parameter.validator
        if validator is None:
            continue
        value = getattr(values, parameter.name)
        try:
            validator(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: List[str] | None = None) -> int:
    base_parser = _build_base_parser()
    args, remaining = base_parser.parse_known_args(argv)

    if args.list:
        _list_processes()
        return 0

    if args.process is None:
        base_parser.print_help(sys.stderr)
        return 1

    _validate_paths(args.paths, base_parser)

    definition = AVAILABLE_PROCESSES[args.process]
    process_parser = _build_process_parser(definition)
    process_args = process_parser.parse_args(remaining)
    try:
        _apply_validators(process_args, list(definition.parameters))
    except argparse.ArgumentTypeError as exc:
        process_parser.error(str(exc))

    options = {parameter.name: getattr(process_args, parameter.name) for parameter in definition.parameters}

    try:
        result = definition.simulate(
            n_paths=args.paths,
            seed=args.seed,
            **options,
        )
    except ValueError as exc:
        base_parser.error(str(exc))

    plot_simulation(result, output=args.output, show=args.show)

    summary = result.parameter_summary()
    endpoints = result.paths[:, -1]
    mean_endpoint = float(np.mean(endpoints))
    std_endpoint = float(np.std(endpoints))
    print(
        f"Simulated {definition.title} with {args.paths} path(s). "
        f"Final value mean={mean_endpoint:.3f}, std={std_endpoint:.3f}."
    )
    if summary:
        print(f"Parameters: {summary}")
    if args.output:
        print(f"Plot written to {args.output}")
    elif not args.show:
        print("No output file requested; use --output to save the figure or --show to display it.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

