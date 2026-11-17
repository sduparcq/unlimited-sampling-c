from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class OGPreparationParameters:
    Np: int


@dataclass
class OGParameters:
    Np: int


@dataclass
class OGPostProcessParameters:
    pass


@dataclass
class OGSolveResult:
    solution: jnp.ndarray
    parameters: OGParameters


def centered_modulo(x, L):
    return x - 2.0 * L * jnp.floor(x / (2.0 * L) + 0.5)


def round_domain(x, L):
    return 2.0 * L * jnp.round(x / (2.0 * L))


def _mod_sample(x, L):
    return x - 2.0 * L * jnp.round(x / (2.0 * L))


def _beta_g(samples, L):
    return 2.0 * L * jnp.ceil(jnp.max(jnp.abs(samples)) / (2.0 * L))


def _compute_kappa(egN, J, beta_g):
    s1 = jnp.cumsum(
        jnp.concatenate(
            (
                jnp.zeros(1, dtype=egN.dtype),
                egN,
            )
        )
    )

    s2 = jnp.cumsum(
        jnp.concatenate(
            (
                jnp.zeros(1, dtype=egN.dtype),
                s1,
            )
        )
    )

    J = jnp.clip(
        J,
        0,
        s2.shape[0] - 1,
    )

    s2_J = s2[int(J)]

    return jnp.floor((s2[1] - s2_J) / (12.0 * beta_g) + 0.5)


def _integrate(x):
    return jnp.cumsum(
        jnp.concatenate(
            (
                jnp.zeros(1, dtype=x.dtype),
                x,
            )
        )
    )


def _round_integrated(x, L):
    return 2.0 * L * jnp.ceil(jnp.floor(x / L) / 2.0)


@dataclass
class OGResult:
    recovered_signal: jnp.ndarray
    eps: jnp.ndarray
    jumps: jnp.ndarray
    solve_result: OGSolveResult
    parameters: OGParameters
    post_process_parameters: OGPostProcessParameters


class PreparedOG:
    def __init__(self, N):
        self.N = N
        self.OG = True

    @staticmethod
    def _solve_stat(
        signal,
        nms,
        N,
        L,
    ):
        beta = _beta_g(
            signal,
            L,
        )

        J = int(jnp.floor(6.0 * beta / L))

        current = nms

        for _ in range(N):
            current = jnp.diff(current)

        dNy = current

        dNepsg = (
            _mod_sample(
                dNy,
                L,
            )
            - dNy
        )

        current = dNepsg

        for _ in range(N - 1):
            k = _compute_kappa(
                current,
                J,
                beta,
            )

            integrated = _integrate(current)

            integrated_rounded = _round_integrated(
                integrated,
                L,
            )

            constant = k * 2.0 * L

            current = integrated_rounded + constant

        return _integrate(current)

    def solve(
        self,
        signal,
        params: OGParameters,
    ):
        eps = PreparedOG._solve_stat(
            signal=signal.signal,
            nms=signal.nms,
            N=self.N,
            L=signal.signal_parameters.L,
        )
        import matplotlib.pyplot as plt

        return OGSolveResult(solution=eps, parameters=params)


class OG:
    def prepare(
        self,
        N: int,
        params: OGPreparationParameters,
    ):
        return PreparedOG(params.Np)

    def post_process(
        self,
        nms,
        solve_result: OGSolveResult,
        params: OGPostProcessParameters,
    ):

        epsg_rec = solve_result.solution

        epsg_rec = epsg_rec - epsg_rec[0]

        jumps = jnp.concatenate(
            (
                jnp.zeros(
                    1,
                    dtype=epsg_rec.dtype,
                ),
                jnp.diff(epsg_rec),
            )
        )

        recovered_signal = nms + epsg_rec

        return OGResult(
            recovered_signal=recovered_signal,
            eps=epsg_rec,
            jumps=jumps,
            solve_result=solve_result,
            parameters=solve_result.parameters,
            post_process_parameters=OGPostProcessParameters(),
        )
