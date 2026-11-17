import time
import numpy as np
from src.computation.gen_lasso import compute_gl

from ...experiments.runner import run_batch
from ...experiments.sweeps import og_sweep
from ...methods import OG
from ...results.og import og_result_to_row
from ...metrics.og import compute_og_metrics

"""
This file aims to fit the parameters of fista
which is the value of tau in the functional
"""


def main():
    experiments = og_sweep(
        # signal params
        M_values=[0.05],
        Omega=10.0,
        bounds=(0.0, 1.0),
        L_values=[0.1],
        sigma_values=[i / 600 for i in range(21)],
        s_seeds=range(1, 101),
        n_seeds=range(101),
        pt=False,
        # mehtod params
        Np_values=[1, 2, 3, 4, 5],
    )

    print(f"Number of experiments: {len(experiments)}")

    start = time.perf_counter()
    run_batch(
        experiments=experiments,
        method=OG(),
        compute_metrics=compute_og_metrics,
        result_to_row=og_result_to_row,
        output_path="results/og/perf/L=0.1-for-comparison-w-dms.csv",
    )

    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
