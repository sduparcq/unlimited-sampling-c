import time

from ..experiments.runner import run_fista_batch
from ..experiments.sweeps import fista_sweep


def main():
    experiments = fista_sweep(
        M=0.05,
        Omega=10.0,
        bounds=(0.0, 1.0),
        L=0.3,
        sigma_values=[0.0, 0.03, 0.05],
        tau_values=[0.05, 0.1],
        threshold_values=[0.5, 0.7],
        max_iter=1000,
        delta=1,
        s_seeds=range(1, 11),
        n_seeds=range(1, 11),
    )

    print(f"Number of experiments: {len(experiments)}")

    start = time.perf_counter()

    run_fista_batch(experiments, output_path="results/fista/benchmark.csv")

    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
