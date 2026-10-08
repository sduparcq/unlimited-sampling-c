import jax.numpy as jnp

from ..operators import D


def compute_gl(
    DFT,
    N: int,
    omega: float,
    epsilon: float,
    delta: int = 1,
):
    K = int(omega) + 1

    DFT_truncated = DFT[
        K + delta : N - (K + delta - 1),
        :,
    ]

    D_matrix = D(N)

    H = jnp.real(DFT_truncated.conj().T @ DFT_truncated)

    H = 0.5 * (H + H.T)

    if epsilon > 0.0:
        H_inv = jnp.linalg.inv(
            H
            + epsilon
            * jnp.eye(
                N,
                dtype=H.dtype,
            )
        )

        Z = jnp.empty(
            (N, 0),
            dtype=H.dtype,
        )
    else:
        H_inv = jnp.linalg.pinv(H)

        eigenvalues, eigenvectors = jnp.linalg.eigh(H)

        eigenvalues = jnp.asarray(eigenvalues)

        scale = jnp.max(jnp.abs(eigenvalues))

        tol = jnp.where(
            scale == 0,
            1e-12,
            jnp.finfo(jnp.float64).eps * N * scale,
        )

        mask = jnp.abs(eigenvalues) <= tol

        Z = eigenvectors[:, mask]

    return (
        DFT_truncated,
        D_matrix,
        H_inv,
        Z,
    )
