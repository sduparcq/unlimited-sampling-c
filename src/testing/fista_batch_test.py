import jax.numpy as jnp

from ..experiments.runner import run_fista_batch
from ..experiments.sweeps import fista_sweep


"""
This file is intended to test the batching system for
experiences on different sets of params
"""


def main():
    experiments = fista_sweep(
        M=0.05,
        Omega=10.0,
        bounds=(0.0, 1.0),
        L=0.3,
        sigma_values=[0.0, 0.03],
        tau_values=[0.05, 0.1],
        threshold_values=[0.5, 0.7],
        max_iter=100,
        delta=1,
        s_seeds=range(1, 51),
        n_seeds=range(1, 51),
    )

    results = run_fista_batch(experiments)
    print(len(results))

    assert len(results) == 32

    for result in results:
        assert result.experiment in experiments

    print(f"Number of experiments: {len(experiments)}")
    print(f"Number of results: {len(results)}")


if __name__ == "__main__":
    main()
