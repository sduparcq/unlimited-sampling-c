def fista_result_to_row(
    experiment,
    metrics,
):
    return {
        "M": experiment.signal.M,
        "Omega": experiment.signal.Omega,
        "L": experiment.signal.L,
        "sigma": experiment.signal.sigma,
        "s_seed": experiment.signal.s_seed,
        "n_seed": experiment.signal.n_seed,
        "delta": experiment.preparation.delta,
        "omega": experiment.preparation.omega,
        "Te": experiment.preparation.Te,
        "tau": experiment.solver.tau,
        "max_iter": experiment.solver.max_iter,
        "threshold": experiment.post_process.threshold,
    }
