import matplotlib.pyplot as plt
import jax.numpy as jnp

from ..methods.dms import (
    DMS,
    DMSParameters,
    DMSPostProcessParameters,
)
from ..signals.factory import SignalFactory
from ..signals.signal import SignalParameters


"""
This file allows to test the algos on an example
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
        sigma=0.0,
        bounds=bounds,
        s_seed=1,
        n_seed=2,
        pt=False,
    )

    factory = SignalFactory()

    signal = factory.generate_signal(signal_params)

    dms = DMS()

    prepared = dms.prepare(N=N)

    solve_params = DMSParameters(
        ext_iter=100,
        int_iter=100,
        alpha=0.015,
        beta=0.18,
    )

    solve_result = prepared.solve(
        z=signal.mod_samples,
        params=solve_params,
    )
    plt.plot(solve_result.x)
    plt.show()

    plt.plot(solve_result.e)
    plt.show()

    post_process_params = DMSPostProcessParameters(
        L=0.3,
        threshold=0.5,
    )

    result = dms.post_process(
        z=signal.mod_samples,
        solve_result=solve_result,
        params=post_process_params,
        D=prepared.D,
    )

    print("N =", N)
    print("Te =", Te)
    print("D shape =", prepared.D.shape)

    print(
        "x finite =",
        jnp.all(jnp.isfinite(solve_result.x)),
    )

    print(
        "e finite =",
        jnp.all(jnp.isfinite(solve_result.e)),
    )

    print(
        "recovered finite =",
        jnp.all(jnp.isfinite(result.recovered_signal)),
    )

    print(
        "jumps =",
        result.jumps,
    )

    t = signal.support

    plt.figure(figsize=(10, 5))

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
        result.recovered_signal,
        "--",
        label="DMS reconstruction",
    )

    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title("DMS reconstruction")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
