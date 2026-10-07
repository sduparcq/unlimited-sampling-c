import time
import numpy as np

from ..experiments.runner import run_batch
from ..experiments.sweeps import fista_sweep
from ..methods import Fista
from ..results.fista import fista_result_to_row
from ..metrics.fista import compute_fista_metrics


def main():
    experiments = fista_sweep(
        # signal params
        M=0.05,
        Omega=10.0,
        bounds=(0.0, 1.0),
        L=0.3,
        sigma_values=[0.0, 0.03, 0.05],
        s_seeds=range(1, 3),
        n_seeds=range(1, 3),
        # method params
        tau_values=[0.05, 0.1],
        max_iter=1000,
        delta=1,
        # post processing params
        threshold_values=[0.1],
    )

    print(f"Number of experiments: {len(experiments)}")

    start = time.perf_counter()
    run_batch(
        experiments=experiments,
        method=Fista(),
        compute_metrics=compute_fista_metrics,
        result_to_row=fista_result_to_row,
        output_path="results/fista/benchmark.csv",
    )

    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
