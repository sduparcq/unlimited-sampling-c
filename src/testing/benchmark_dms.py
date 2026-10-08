import time

from ..experiments.runner import run_batch
from ..experiments.sweeps import dms_sweep
from ..methods import DMS
from ..results.dms import dms_result_to_row
from ..metrics.dms import compute_dms_metrics


def main():
    experiments = dms_sweep(
        # signal params
        M=0.05,
        Omega=10.0,
        bounds=(0.0, 1.0),
        L=0.3,
        sigma_values=[0.0, 0.03, 0.05],
        s_seeds=range(1, 6),
        n_seeds=range(1, 6),
        pt=False,
        # mehtod params
        alpha_values=[0.001, 0.1],
        beta_values=[0.1, 1],
        ext_iter=100,
        int_iter=10,
        # post process params
        threshold_values=[0.3, 0.5, 0.7, 0.9],
    )

    print(f"Number of experiments: {len(experiments)}")

    start = time.perf_counter()
    run_batch(
        experiments=experiments,
        method=DMS(),
        compute_metrics=compute_dms_metrics,
        result_to_row=dms_result_to_row,
        output_path="results/dms/benchmark.csv",
    )

    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
