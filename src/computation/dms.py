import jax.numpy as jnp

from ..operators import D


def compute_dms(N: int):
    D_matrix = D(N)
    I = jnp.eye(N)

    return D_matrix, I
