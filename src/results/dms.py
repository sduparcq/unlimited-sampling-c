def dms_result_to_row(
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
        "ext_iter": experiment.solver.ext_iter,
        "int_iter": experiment.solver.int_iter,
        "alpha": experiment.solver.alpha,
        "beta": experiment.solver.beta,
        # post process params
        "threshold": experiment.post_process.threshold,
        # metrics
        "is_recovered": metrics.is_recovered,
        "uot_distance_q": metrics.uot_distance_q,
    }
