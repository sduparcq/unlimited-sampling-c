def gl_result_to_row(
    experiment,
    metrics,
):
    return {
        # signal params
        "M": experiment.signal.M,
        "Omega": experiment.signal.Omega,
        "L": experiment.signal.L,
        "sigma": experiment.signal.sigma,
        "s_seed": experiment.signal.s_seed,
        "n_seed": experiment.signal.n_seed,
        # algo params
        "delta": experiment.preparation.delta,
        "tau": experiment.solver.tau,
        "lin": experiment.solver.lin,
        # post process params
        "threshold": experiment.post_process.threshold,
        # metrics
        "is_recovered": metrics.is_recovered,
        "uot_distance": metrics.uot_distance,
        "uot_distance_q": metrics.uot_distance_q,
    }
