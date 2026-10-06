import jax.numpy as jnp


def compute_fista(
    DFT,
    D_inv,
    N: int,
    delta: int,
    omega: float,
    Te: float,
):
    freq_step = 2.0 * jnp.pi / (N * Te)

    K = int(jnp.ceil(omega / freq_step))

    DFT_truncated = DFT[
        K + delta : N - (K + delta - 1),
        :,
    ]

    A = DFT_truncated @ D_inv

    lip = float(
        jnp.linalg.norm(
            A,
            ord=2,
        )
        ** 2
    )

    return DFT_truncated, lip
