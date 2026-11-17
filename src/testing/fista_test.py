import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from ..methods.fista import (
    Fista,
    FistaParameters,
    FistaPostProcessParameters,
    FistaPreparationParameters,
)

from ..metrics import *
from ..signals.factory import SignalFactory
from ..signals.signal import SignalParameters


"""
This file allows to test the algos on an example signal
"""


def main():
    M = 0.05
    Omega = 10.0
    bounds = (0.0, 1.0)

    Te = (jnp.pi / Omega) * M

    N = int(jnp.ceil(1.0 + (bounds[1] - bounds[0]) / Te))

    signal_params = SignalParameters(
        N=N,
        M=M,
        Omega=Omega,
        L=0.3,
        sigma=0.09,
        bounds=bounds,
        s_seed=3,
        n_seed=20,
        pt=True,
    )

    factory = SignalFactory()
    signal = factory.generate_signal(signal_params)

    fista = Fista()
    prepare_params = FistaPreparationParameters(
        delta=1,
        omega=Omega,
        Te=Te,
    )
    prepared = fista.prepare(
        params=prepare_params,
        N=N,
    )

    solve_params = FistaParameters(
        tau=0.1,
        max_iter=10_000,
    )

    solve_result = prepared.solve(
        nms=signal.nms,
        params=solve_params,
    )

    post_process_params = FistaPostProcessParameters(
        L=0.3,
        threshold=0.9,
    )

    result = fista.post_process(
        nms=signal.nms,
        solve_result=solve_result,
        params=post_process_params,
    )

    metrics = compute_fista_metrics(signal=signal, result=result)
    print(f"metrics {metrics}")
    t = signal.support

    plt.subplot(2, 1, 1)
    plt.plot(
        t,
        signal.signal,
        label="Signal",
    )

    plt.plot(
        t,
        signal.mod_samples,
        label="Modulo samples",
    )

    plt.plot(
        t,
        result.recovered_signal.real,
        "--",
        label="FISTA reconstruction",
    )
    plt.subplot(2, 1, 2)
    plt.plot(
        t,
        np.diff(signal.signal - signal.mod_samples, prepend=0),
        label="true jumps",
    )
    plt.plot(
        t,
        result.solve_result.jumps,
        "--",
        label="rec jumps",
    )

    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title("FISTA reconstruction")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
