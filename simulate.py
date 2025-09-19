"""Command line interface for the stochastic process simulator."""

from __future__ import annotations

import argparse
import textwrap
from typing import Dict

from stochastic_processes import PROCESS_REGISTRY, get_process_spec, list_processes, simulate_process
from stochastic_processes.plotting import plot_simulation


def _format_process_listing() -> str:
    lines = []
    for spec in list_processes():
        lines.append(f"- {spec.key}: {spec.name} ({spec.category})")
        lines.append(textwrap.indent(textwrap.fill(spec.description, width=80), prefix="  "))
    return "\n".join(lines)


def _parse_params(spec_key: str, params: list[str]) -> Dict[str, object]:
    spec = get_process_spec(spec_key)
    overrides: Dict[str, object] = {}
    for item in params:
        if "=" not in item:
            raise ValueError(
                f"Could not parse parameter override '{item}'. Expected the form name=value."
            )
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in spec.parameters:
            raise KeyError(f"Unknown parameter '{name}' for process '{spec_key}'.")
        overrides[name] = spec.parameters[name].parse(raw_value.strip())
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and plot discrete and continuous time stochastic processes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available processes:\n" + _format_process_listing(),
    )
    parser.add_argument(
        "--process",
        required=True,
        choices=sorted(PROCESS_REGISTRY.keys()),
        help="Identifier of the process to simulate. Use the epilog list for options.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="name=value",
        help="Override one of the process parameters (can be provided multiple times).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator to obtain reproducible simulations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path of the image file where the plot should be stored. If omitted the plot is shown.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display of the plot (useful when saving to a file).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = _parse_params(args.process, args.param)
    result = simulate_process(args.process, overrides=overrides, seed=args.seed)
    spec = get_process_spec(args.process)

    plot_simulation(
        result,
        spec,
        output_path=args.output,
        show=not args.no_show and args.output is None,
    )


if __name__ == "__main__":
    main()
