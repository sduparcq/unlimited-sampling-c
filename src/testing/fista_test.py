import matplotlib.pyplot as plt
import jax.numpy as jnp

from ..methods.fista import (
    Fista,
    FistaParameters,
)
from ..signals.factory import SignalFactory
from ..signals.signal import SignalParameters


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

    fista = Fista()

    prepared = fista.prepare(
        N=N,
        delta=1,
        omega=Omega,
        Te=Te,
    )

    params = FistaParameters(
        tau=0.1,
        max_iter=10_000,
        delta=1,
    )

    result = prepared.solve(
        y_mod=signal.mod_samples,
        params=params,
    )

    print("N:", N)
    print("Te:", Te)
    print(
        "DFT shape:",
        prepared.DFT_truncated.shape,
    )
    print(
        "Lipschitz:",
        prepared.lip,
    )
    print(
        "recovered min:",
        jnp.min(result.recovered_signal.real),
    )
    print(
        "recovered max:",
        jnp.max(result.recovered_signal.real),
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
        result.recovered_signal.real,
        "--",
        label="FISTA reconstruction",
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
