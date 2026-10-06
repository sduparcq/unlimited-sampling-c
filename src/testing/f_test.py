from ..experiments.sweeps import fista_sweep

experiments = fista_sweep(
    M=0.05,
    Omega=10.0,
    bounds=(0.0, 1.0),
    L=0.3,
    sigma_values=[0.0, 0.03, 0.05, 0.07, 0.09],
    tau_values=[0.05, 0.1, 0.15],
    threshold_values=[0.5, 0.6, 0.7, 0.8],
    max_iter=10_000,
    delta=1,
    s_seeds=range(1, 4),
    n_seeds=range(1, 4),
)

print(experiments)
