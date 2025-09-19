# Stochastic Processes Simulator

A small command-line tool for simulating and visualising common discrete-time
and continuous-time stochastic processes. The simulator can generate one or
multiple sample paths, display them using Matplotlib and/or save the output as
an image.

## Getting started

Install the Python dependencies (Python 3.10+ is recommended):

```bash
pip install -r requirements.txt
```

## Usage

List the available processes:

```bash
python simulate.py --list
```

Simulate a process by providing its identifier and any optional parameters. All
process-specific options can be inspected via `--help` after the process name.
For example, to simulate a random walk with 300 steps and save the resulting
plot:

```bash
python simulate.py random_walk --steps 300 --paths 3 --output random_walk.png
```

To inspect the configuration options for the Ornstein-Uhlenbeck process:

```bash
python simulate.py ornstein_uhlenbeck --help
```

By default the command prints a short numerical summary of the simulated sample
paths. Use `--show` to display the plot in an interactive window (requires a
matplotlib backend with GUI support) or `--output <file>` to save the plot.

## Included processes

The following processes are currently implemented:

| Identifier | Category | Description |
| ---------- | -------- | ----------- |
| `random_walk` | Discrete-time | Simple random walk with configurable step probability and magnitude. |
| `ar1` | Discrete-time | Gaussian autoregressive process of order one. |
| `brownian_motion` | Continuous-time | Brownian motion with optional drift and volatility adjustments. |
| `geometric_brownian_motion` | Continuous-time | Geometric Brownian motion, a common financial diffusion. |
| `poisson_process` | Continuous-time | Homogeneous Poisson counting process. |
| `ornstein_uhlenbeck` | Continuous-time | Mean-reverting Ornstein-Uhlenbeck diffusion. |

Each process exposes intuitive parameters such as the number of steps, drift,
volatility, intensity, or initial conditions. Refer to the per-process help for
the exact defaults.

## Development

The code resides in the `stochastic_processes` package and is orchestrated by
`simulate.py`. Running `python -m compileall .` ensures that all modules are
syntactically valid.
