import jax.numpy as jnp


D_cache = {}
D_inv_cache = {}


def D(N: int):
    if N in D_cache:
        return D_cache[N]
    else:
        D = jnp.zeros((N - 1, N))
        D = D.at[jnp.arange(N - 1), jnp.arange(N - 1)].set(-1)
        D = D.at[jnp.arange(N - 1), jnp.arange(1, N)].set(1)

        D_cache[N] = D
        return D


def D_inv(N: int):
    if N not in D_inv_cache:
        D_inv_cache[N] = jnp.tril(jnp.ones((N, N)))
    return D_inv_cache[N]
