from itertools import product

import jax.numpy as jnp

from ..methods.dms import (
    DMSParameters,
    DMSPostProcessParameters,
    DMSPreparationParameters,
)
from ..methods.fista import (
    FistaParameters,
    FistaPostProcessParameters,
    FistaPreparationParameters,
)
from ..signals.signal import SignalParameters
from .experiments import DMSExperiment, FistaExperiment


def fista_sweep(
    *,
    M,
    Omega,
    bounds,
    L,
    sigma_values,
    tau_values,
    threshold_values,
    max_iter,
    delta,
    s_seeds,
    n_seeds,
    pt=False,
):
    Te = (jnp.pi / Omega) * M

    N = int(jnp.ceil(1.0 + (bounds[1] - bounds[0]) / Te))

    experiments = []

    for (
        sigma,
        tau,
        threshold,
        s_seed,
        n_seed,
    ) in product(
        sigma_values,
        tau_values,
        threshold_values,
        s_seeds,
        n_seeds,
    ):
        signal_parameters = SignalParameters(
            N=N,
            M=M,
            Omega=Omega,
            L=L,
            sigma=sigma,
            bounds=bounds,
            s_seed=s_seed,
            n_seed=n_seed,
            pt=pt,
        )

        preparation_parameters = FistaPreparationParameters(
            delta=delta,
            omega=Omega,
            Te=Te,
        )

        solver_parameters = FistaParameters(
            tau=tau,
            max_iter=max_iter,
        )

        post_process_parameters = FistaPostProcessParameters(
            L=L,
            threshold=threshold,
        )

        experiments.append(
            FistaExperiment(
                signal=signal_parameters,
                preparation=preparation_parameters,
                solver=solver_parameters,
                post_process=post_process_parameters,
            )
        )

    return experiments


def dms_sweep(
    *,
    M,
    Omega,
    bounds,
    L,
    sigma_values,
    alpha_values,
    beta_values,
    threshold_values,
    ext_iter,
    int_iter,
    s_seeds,
    n_seeds,
    pt=False,
):
    Te = (jnp.pi / Omega) * M

    N = int(jnp.ceil(1.0 + (bounds[1] - bounds[0]) / Te))

    experiments = []

    for (
        sigma,
        alpha,
        beta,
        threshold,
        s_seed,
        n_seed,
    ) in product(
        sigma_values,
        alpha_values,
        beta_values,
        threshold_values,
        s_seeds,
        n_seeds,
    ):
        signal_parameters = SignalParameters(
            N=N,
            M=M,
            Omega=Omega,
            L=L,
            sigma=sigma,
            bounds=bounds,
            s_seed=s_seed,
            n_seed=n_seed,
            pt=pt,
        )

        preparation_parameters = DMSPreparationParameters()

        solver_parameters = DMSParameters(
            ext_iter=ext_iter,
            int_iter=int_iter,
            alpha=alpha,
            beta=beta,
        )

        post_process_parameters = DMSPostProcessParameters(
            L=L,
            threshold=threshold,
        )

        experiments.append(
            DMSExperiment(
                signal=signal_parameters,
                preparation=preparation_parameters,
                solver=solver_parameters,
                post_process=post_process_parameters,
            )
        )

    return experiments
