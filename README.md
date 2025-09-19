# Stochastic-Prossesses

A lightweight simulator and plotter for common discrete- and continuous-time
stochastic processes. The command line interface lets you pick a process,
customise its parameters, and either display or save the simulated sample
paths.

## Features

- Discrete-time processes: simple random walk, autoregressive AR(1) process,
  and a two-state Markov chain.
- Continuous-time processes: Poisson process, Brownian motion, geometric
  Brownian motion, and Ornstein-Uhlenbeck diffusion.
- Configurable number of steps, time horizon, distribution parameters, and
  number of simulated sample paths.
- Generates publication-ready Matplotlib graphs that can be shown
  interactively or exported to an image file.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **List available processes**

   ```bash
   python simulate.py --process simple_random_walk --help
   ```

   The help output includes an epilog that lists every process with a short
   description and shows the parameters that can be overridden via the
   `--param` flag.

3. **Run a simulation**

   Override any parameter by using the `--param name=value` flag (it can be
   supplied multiple times). Example: simulate an Ornstein-Uhlenbeck process
   and save the output plot to `ou.png`.

   ```bash
   python simulate.py --process ornstein_uhlenbeck \
       --param theta=1.5 --param sigma=0.5 --param paths=10 \
       --output ou.png --no-show
   ```

   To simply visualise a process interactively, omit the `--output` flag and
   the plot window will open after the simulation finishes.

## Development

The simulator is organised as a small Python package inside the
`stochastic_processes` directory. The most important modules are:

- `discrete.py` and `continuous.py`: implementations of each process.
- `registry.py`: metadata describing the processes and their parameters.
- `plotting.py`: helper to generate Matplotlib figures from a simulation.

You can import the package in your own Python code to reuse the simulations:

```python
from stochastic_processes import simulate_process

result = simulate_process("poisson_process", {"rate": 3.0, "paths": 2})
print(result.time)
print(result.values)
```

Run the tests to ensure everything works as expected:

```bash
pytest
```
