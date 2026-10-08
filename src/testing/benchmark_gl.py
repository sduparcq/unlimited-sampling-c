import time
import numpy as np

from ..experiments.runner import run_batch
from ..experiments.sweeps import gen_lasso_sweep
from ..methods import GenLasso
from ..results.gen_lasso import gl_result_to_row
from ..metrics.gen_lasso import compute_gen_lasso_metrics


def main():
    experiments = gen_lasso_sweep(
        # signal params
        M=0.05,
        Omega=10.0,
        bounds=(0.0, 1.0),
        L=0.3,
        sigma_values=[0.0, 0.03],
        s_seeds=range(1, 6),
        n_seeds=range(1, 6),
        pt=False,
        # method params
        tau_values=[0.01],
        lin_values=[True],
        solver="CLARABEL",
        epsilon=0.0,
        # post processing params
        threshold_values=[0.4],
    )

    print(f"Number of experiments: {len(experiments)}")

    start = time.perf_counter()

    run_batch(
        experiments=experiments,
        method=GenLasso(),
        compute_metrics=compute_gen_lasso_metrics,
        result_to_row=gl_result_to_row,
        output_path="results/gen_lasso/benchmark.csv",
    )

    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
