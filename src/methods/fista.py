from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..computation.fista import compute_fista
from ..operators import D_inv, DFT_matrix


jax.config.update("jax_enable_x64", True)


@jax.jit
def soft_threshold_jax(x, gamma):
    return jnp.sign(x) * jnp.maximum(
        jnp.abs(x) - gamma,
        0.0,
    )


@jax.jit
def lambda_ise_jax(x, L, threshold):
    return jnp.where(
        x > threshold * L,
        2.0 * L,
        jnp.where(
            x < -threshold * L,
            -2.0 * L,
            0.0,
        ),
    )


@dataclass
class FistaPreparationParameters:
    delta: int
    omega: float
    Te: float


@dataclass
class FistaParameters:
    tau: float
    max_iter: int


@dataclass
class FistaPostProcessParameters:
    L: float
    threshold: float


@dataclass
class FistaSolveResult:
    jumps: jnp.ndarray
    parameters: FistaParameters


@dataclass
class FistaResult:
    recovered_signal: jnp.ndarray
    eps: jnp.ndarray
    jumps: jnp.ndarray
    solve_result: FistaSolveResult
    parameters: FistaParameters
    post_process_parameters: FistaPostProcessParameters


class PreparedFista:
    def __init__(
        self,
        DFT_truncated: jnp.ndarray,
        D_inv: jnp.ndarray,
        lip: float,
    ):
        self.DFT_truncated = DFT_truncated
        self.D_inv = D_inv
        self.lip = lip

    @staticmethod
    def _solve_optim(
        b,
        DFT_truncated,
        max_iter,
        tau,
        lip,
    ):
        N = DFT_truncated.shape[1]
        gamma = 1.0 / lip

        x = jnp.zeros(
            N,
            dtype=DFT_truncated.dtype,
        )

        x_prev = x
        y = x
        s = 1.0

        def body(_, state):
            x, x_prev, y, s = state

            D_inv_y = jnp.cumsum(y)

            residual = DFT_truncated @ D_inv_y - b

            grad_freq = DFT_truncated.conj().T @ residual

            grad = jnp.cumsum(grad_freq[::-1])[::-1]

            z = y - gamma * grad

            x_new = soft_threshold_jax(
                z,
                tau * gamma,
            )

            s_new = (1.0 + jnp.sqrt(1.0 + 4.0 * s**2)) / 2.0

            beta = (s - 1.0) / s_new

            y_new = x_new + beta * (x_new - x)

            return (
                x_new,
                x,
                y_new,
                s_new,
            )

        x, _, _, _ = jax.lax.fori_loop(
            0,
            max_iter,
            body,
            (
                x,
                x_prev,
                y,
                s,
            ),
        )

        return x.real

    def solve(
        self,
        y_mod,
        params: FistaParameters,
    ):
        y_mod = jnp.asarray(
            y_mod,
            dtype=jnp.complex128,
        )

        b = -(self.DFT_truncated @ y_mod)

        jumps = self._solve_optim(
            b=b,
            DFT_truncated=self.DFT_truncated,
            max_iter=params.max_iter,
            tau=params.tau,
            lip=self.lip,
        )

        return FistaSolveResult(
            jumps=jumps,
            parameters=params,
        )


class Fista:
    def prepare(
        self,
        params: FistaPreparationParameters,
        N: int,
    ):
        D_inv_matrix = D_inv(N)
        DFT = DFT_matrix(N)

        DFT_truncated, lip = compute_fista(
            DFT=DFT,
            D_inv=D_inv_matrix,
            N=N,
            delta=params.delta,
            omega=params.omega,
            Te=params.Te,
        )

        return PreparedFista(
            DFT_truncated=DFT_truncated,
            D_inv=D_inv_matrix,
            lip=lip,
        )

    def post_process(
        self,
        y_mod,
        solve_result: FistaSolveResult,
        params: FistaPostProcessParameters,
    ):
        y_mod = jnp.asarray(
            y_mod,
            dtype=jnp.complex128,
        )

        jumps = lambda_ise_jax(
            solve_result.jumps,
            params.L,
            params.threshold,
        )

        eps = jnp.cumsum(jumps)

        recovered_signal = y_mod + eps

        return FistaResult(
            recovered_signal=recovered_signal,
            eps=eps,
            jumps=jumps,
            solve_result=solve_result,
            parameters=solve_result.parameters,
            post_process_parameters=params,
        )
