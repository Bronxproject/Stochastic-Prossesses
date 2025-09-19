import numpy as np

from stochastic_processes import get_process_spec, simulate_process


def test_random_walk_returns_expected_shape():
    result = simulate_process(
        "simple_random_walk",
        overrides={"steps": 50, "paths": 2, "start": 1.0, "p": 0.6},
        seed=42,
    )
    assert result.values.shape == (2, 51)
    np.testing.assert_allclose(result.values[:, 0], 1.0)


def test_poisson_process_is_non_decreasing():
    result = simulate_process(
        "poisson_process",
        overrides={"rate": 5.0, "t_max": 2.0, "dt": 0.1, "paths": 3},
        seed=123,
    )
    diffs = np.diff(result.values, axis=1)
    assert np.all(diffs >= 0)


def test_geometric_brownian_motion_positive():
    result = simulate_process(
        "geometric_brownian_motion",
        overrides={"mu": 0.05, "sigma": 0.1, "initial": 2.0, "paths": 4},
        seed=99,
    )
    assert np.all(result.values > 0)


def test_default_parameters_are_accessible():
    spec = get_process_spec("ornstein_uhlenbeck")
    params = spec.coerce_parameters()
    assert params["theta"] == spec.parameters["theta"].default
    assert params["paths"] == spec.parameters["paths"].default
