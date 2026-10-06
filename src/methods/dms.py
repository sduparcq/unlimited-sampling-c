from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..computation.dms import compute_dms


@jax.jit
def tilde_psi(x, alpha, beta):
    a = alpha / (2.0 * beta)

    return jnp.where(
        x == 0,
        0.0,
        jnp.where(
            x <= a,
            x,
            alpha / beta * (1.0 - alpha / (4.0 * beta * x)),
        ),
    )


@jax.jit
def project_jumps(
    z,
    e,
    L,
    threshold,
    D,
):
    e_t = (e > threshold).astype(jnp.int32)

    a = -jnp.sign(D @ z)

    return 2.0 * L * a * e_t


@jax.jit
def lambda_ise_jax(x, L):
    return 2.0 * L * jnp.round(x / (2.0 * L))


@dataclass
class DMSPreparationParameters:
    pass


@dataclass
class DMSParameters:
    ext_iter: int
    int_iter: int
    alpha: float
    beta: float


@dataclass
class DMSPostProcessParameters:
    L: float
    threshold: float


@dataclass
class DMSSolveResult:
    x: jnp.ndarray
    e: jnp.ndarray
    parameters: DMSParameters


@dataclass
class DMSResult:
    recovered_signal: jnp.ndarray
    eps: jnp.ndarray
    jumps: jnp.ndarray
    solve_result: DMSSolveResult
    parameters: DMSParameters
    post_process_parameters: DMSPostProcessParameters


class PreparedDMS:
    def __init__(
        self,
        D: jnp.ndarray,
        I: jnp.ndarray,
    ):
        self.D = D
        self.I = I

    @staticmethod
    def _solve_optim(
        z,
        D,
        I,
        params,
        gamma=0.1,
    ):
        def f(x):
            return 0.5 * jnp.sum((x - z) ** 2)

        df = jax.grad(f)

        d_t_psi = jax.grad(
            lambda x: tilde_psi(
                x,
                params.alpha,
                params.beta,
            )
        )

        x = z.copy()

        def prox_grad(
            x,
            lambda_vec,
        ):
            u = x - gamma * df(x)

            LAMBDA = jnp.diag(lambda_vec)

            A = I + 2.0 * gamma * params.beta * D.T @ LAMBDA @ D

            return jnp.linalg.solve(
                A,
                u,
            )

        for _ in range(params.ext_iter):
            dx = D @ x

            lambda_vec = jax.vmap(d_t_psi)(dx**2)

            new_x = x

            for _ in range(params.int_iter):
                new_x = prox_grad(
                    new_x,
                    lambda_vec,
                )

            if jnp.linalg.norm(
                new_x - x,
                ord=2,
            ) < 10 ** (-5):
                x = new_x
                break

            x = new_x

        Dx = D @ x

        delta = params.alpha / (params.beta * (jnp.abs(Dx) ** 2))

        e = jnp.maximum(
            1.0 - delta,
            0.0,
        )

        return x, e

    def solve(
        self,
        z,
        params: DMSParameters,
    ):
        z = jnp.asarray(z)

        x, e = self._solve_optim(
            z=z,
            D=self.D,
            I=self.I,
            params=params,
        )

        return DMSSolveResult(
            x=x,
            e=e,
            parameters=params,
        )


class DMS:
    def prepare(
        self,
        params: DMSPreparationParameters,
        N: int,
    ):
        D_mat, I = compute_dms(N)

        return PreparedDMS(
            D=D_mat,
            I=I,
        )

    def post_process(
        self,
        z,
        solve_result: DMSSolveResult,
        params: DMSPostProcessParameters,
        D,
    ):
        z = jnp.asarray(z)

        jumps = project_jumps(
            z=z,
            e=solve_result.e,
            L=params.L,
            threshold=params.threshold,
            D=D,
        )

        jumps = jnp.concatenate(
            [
                jnp.array([0.0]),
                jumps,
            ]
        )

        eps = jnp.cumsum(jumps)

        recovered_signal = z + eps

        return DMSResult(
            recovered_signal=recovered_signal,
            eps=eps,
            jumps=jumps,
            solve_result=solve_result,
            parameters=solve_result.parameters,
            post_process_parameters=params,
        )
