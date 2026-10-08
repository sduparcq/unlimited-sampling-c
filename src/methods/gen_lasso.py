from dataclasses import dataclass

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from ..computation.gen_lasso import compute_gl
from ..operators import DFT_matrix


jax.config.update("jax_enable_x64", True)


@dataclass
class GLPreparationParameters:
    omega: float
    delta: int = 1
    epsilon: float = 0.0


@dataclass
class GLParameters:
    tau: float
    lin: bool
    solver: str = "CLARABEL"


@dataclass
class GLPostProcessParameters:
    L: float
    threshold: float


@dataclass
class GLSolveResult:
    beta: jnp.ndarray
    beta_minimal: jnp.ndarray
    jumps: jnp.ndarray
    u_star: jnp.ndarray
    parameters: GLParameters


@dataclass
class GLResult:
    recovered_signal: jnp.ndarray
    eps: jnp.ndarray
    jumps: jnp.ndarray
    solve_result: GLSolveResult
    parameters: GLParameters
    post_process_parameters: GLPostProcessParameters


@jax.jit
def _compute_quadratic_terms(A, b):
    H = jnp.real(A.conj().T @ A)
    H = 0.5 * (H + H.T)
    g = jnp.real(A.conj().T @ b)
    return H, g


@jax.jit
def _compute_pseudoinverse(H):
    return jnp.linalg.pinv(H)


@jax.jit
def _compute_regularized_inverse(H, epsilon):
    H_reg = H + epsilon * jnp.eye(
        H.shape[0],
        dtype=H.dtype,
    )
    return jnp.linalg.inv(H_reg)


@jax.jit
def _compute_nullspace_eigendecomposition(H):
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    return eigenvalues, eigenvectors


@jax.jit
def _compute_dual_matrices(D, H_inv, g):
    Q1 = D @ H_inv @ D.T
    Q1 = 0.5 * (Q1 + Q1.T)
    Q2 = D @ H_inv @ g
    return Q1, Q2


@jax.jit
def _compute_beta_0(H_inv, g, D, u_star):
    return H_inv @ (g - D.T @ u_star)


def _get_nullspace(H):
    eigenvalues, eigenvectors = _compute_nullspace_eigendecomposition(H)

    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)

    scale = np.max(np.abs(eigenvalues))

    if scale == 0:
        tol = 1e-12
    else:
        tol = np.finfo(np.float64).eps * H.shape[0] * scale

    mask = np.abs(eigenvalues) <= tol

    return eigenvectors[:, mask]


def generalized_lasso_dual(
    A,
    b,
    D,
    tau,
    H_inv=None,
    Z=None,
    solver="CLARABEL",
    epsilon=0.0,
):
    A = jnp.asarray(A)
    b = jnp.asarray(b)

    D = jnp.asarray(
        D,
        dtype=jnp.float64,
    )

    if tau < 0:
        raise ValueError("tau must be non-negative.")

    m, N = A.shape
    p, N_D = D.shape

    if b.ndim != 1 or b.shape[0] != m:
        raise ValueError(f"b must have shape ({m},), got {b.shape}")

    if N_D != N:
        raise ValueError(f"D must have {N} columns, got {N_D}")

    H, g = _compute_quadratic_terms(
        A,
        b,
    )

    if epsilon > 0.0:
        if H_inv is None:
            H_inv = _compute_regularized_inverse(
                H,
                epsilon,
            )

        if Z is None:
            Z = np.empty(
                (N, 0),
            )
    else:
        if H_inv is None:
            H_inv = _compute_pseudoinverse(H)

        if Z is None:
            Z = _get_nullspace(H)

    Q1, Q2 = _compute_dual_matrices(
        D,
        H_inv,
        g,
    )

    Q1 = np.asarray(Q1)
    Q2 = np.asarray(Q2)
    D = np.asarray(D)

    u = cp.Variable(p)

    constraints = [
        cp.norm_inf(u) <= tau,
    ]

    if Z.shape[1] > 0:
        constraints.append(Z.T @ D.T @ u == 0)

    dual_objective = cp.Minimize(
        0.5
        * cp.quad_form(
            u,
            cp.psd_wrap(Q1),
        )
        - Q2.T @ u
    )

    dual_problem = cp.Problem(
        dual_objective,
        constraints,
    )

    dual_problem.solve(
        solver=solver,
    )

    if u.value is None:
        raise RuntimeError(f"Dual solver failed: {dual_problem.status}")

    return np.asarray(u.value).reshape(-1)


def recover_primal_from_dual_min(
    A,
    b,
    D,
    u_star,
    H_inv=None,
    Z=None,
):
    A = jnp.asarray(A)
    b = jnp.asarray(b)

    D_jnp = jnp.asarray(
        D,
        dtype=jnp.float64,
    )

    u_star = jnp.asarray(
        u_star,
        dtype=jnp.float64,
    )

    H, g = _compute_quadratic_terms(
        A,
        b,
    )

    if H_inv is None:
        H_inv = _compute_pseudoinverse(H)

    if Z is None:
        Z = _get_nullspace(H)

    beta_0 = _compute_beta_0(
        H_inv,
        g,
        D_jnp,
        u_star,
    )

    beta_0 = np.real(np.asarray(beta_0))

    D_np = np.asarray(D)

    if Z.shape[1] == 0:
        return (
            beta_0,
            beta_0,
        )

    c = cp.Variable(Z.shape[1])

    beta_c = beta_0 + np.asarray(Z) @ c

    nullspace_problem = cp.Problem(
        cp.Minimize(cp.norm1(D_np @ beta_c)),
        [beta_c[0] == 0],
    )

    nullspace_problem.solve(solver="CLARABEL")

    if c.value is None:
        raise RuntimeError("Nullspace recovery failed.")

    beta_star = beta_0 + np.asarray(Z) @ c.value

    return (
        np.real(beta_star),
        beta_0,
    )


def recover_primal_from_dual_linear(
    A,
    b,
    D,
    u_star,
    y_mod,
    tau,
    L,
    H_inv=None,
):
    tol = 1e-10

    N = A.shape[1]
    P = A.shape[0]

    S = np.abs(np.asarray(u_star)) >= tau - tol

    Sc = ~S

    D_Sc = D[Sc, :]

    A_jnp = jnp.asarray(A)
    b_jnp = jnp.asarray(b)

    D_jnp = jnp.asarray(
        D,
        dtype=jnp.float64,
    )

    if H_inv is None:
        H = jnp.real(A_jnp.conj().T @ A_jnp)

        H = 0.5 * (H + H.T)

        H_inv = _compute_pseudoinverse(H)

    g = jnp.real(A_jnp.conj().T @ b_jnp)

    beta_s = _compute_beta_0(
        H_inv,
        g,
        D_jnp,
        jnp.asarray(
            u_star,
            dtype=jnp.float64,
        ),
    )

    D_beta_s = D_jnp @ beta_s
    D_beta_s_Sc = D_beta_s[Sc]

    mat = jnp.concatenate(
        [
            jnp.asarray(D_Sc),
            A_jnp,
            jnp.ones((1, N)),
        ],
        axis=0,
    )

    d = jnp.concatenate(
        [
            -D_beta_s_Sc,
            jnp.zeros(P),
            jnp.zeros(1),
        ]
    )

    v = jnp.linalg.pinv(mat) @ d

    beta_star = beta_s + v

    return (
        beta_star,
        beta_s,
    )


def recover_primal_from_dual_cvxpy(
    A,
    b,
    D,
    u_star,
    tau,
    solver="CLARABEL",
):
    N = A.shape[1]

    beta = cp.Variable(
        N,
        complex=True,
    )

    A_np = np.asarray(A)
    b_np = np.asarray(b)
    D_np = np.asarray(D)

    objective = cp.Minimize(0.5 * cp.sum_squares(cp.abs(A_np @ beta - b_np)))

    constraints = [cp.abs(D_np @ beta) <= tau]

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve(
        solver=solver,
    )

    if beta.value is None:
        raise RuntimeError("Primal generalized lasso problem did not converge.")

    return jnp.asarray(beta.value)


class PreparedGenLasso:
    def __init__(
        self,
        DFT_truncated,
        D,
        H_inv,
        Z,
    ):
        self.DFT_truncated = DFT_truncated
        self.D = D
        self.H_inv = H_inv
        self.Z = Z

    def solve(
        self,
        nms,
        params: GLParameters,
    ):
        nms = jnp.asarray(
            nms,
            dtype=jnp.complex128,
        )

        b = -(self.DFT_truncated @ nms)

        u_star = generalized_lasso_dual(
            A=self.DFT_truncated,
            b=b,
            D=self.D,
            tau=params.tau,
            H_inv=self.H_inv,
            Z=self.Z,
            solver=params.solver,
        )

        if params.lin:
            beta, beta_minimal = recover_primal_from_dual_linear(
                A=self.DFT_truncated,
                b=b,
                D=self.D,
                u_star=u_star,
                y_mod=nms,
                tau=params.tau,
                L=0.0,
                H_inv=self.H_inv,
            )
        else:
            beta, beta_minimal = recover_primal_from_dual_min(
                A=self.DFT_truncated,
                b=b,
                D=self.D,
                u_star=u_star,
                H_inv=self.H_inv,
                Z=self.Z,
            )
        jumps = np.diff(beta, prepend=0)
        return GLSolveResult(
            beta=beta,
            beta_minimal=beta_minimal,
            jumps=jumps,
            u_star=u_star,
            parameters=params,
        )


class GenLasso:
    def prepare(
        self,
        params: GLPreparationParameters,
        N: int,
    ):
        DFT = DFT_matrix(N)

        (
            DFT_truncated,
            D_matrix,
            H_inv,
            Z,
        ) = compute_gl(
            DFT=DFT,
            N=N,
            omega=params.omega,
            epsilon=params.epsilon,
            delta=params.delta,
        )

        return PreparedGenLasso(
            DFT_truncated=DFT_truncated,
            D=D_matrix,
            H_inv=H_inv,
            Z=Z,
        )

    def post_process(
        self,
        nms,
        solve_result: GLSolveResult,
        params: GLPostProcessParameters,
    ):
        nms = jnp.asarray(
            nms,
            dtype=jnp.complex128,
        )

        beta = solve_result.beta

        jumps = jnp.concatenate(
            [
                jnp.array([0.0]),
                jnp.diff(beta),
            ]
        )

        eps = jnp.cumsum(jumps)

        recovered_signal = nms + eps

        return GLResult(
            recovered_signal=recovered_signal,
            eps=eps,
            jumps=jumps,
            solve_result=solve_result,
            parameters=solve_result.parameters,
            post_process_parameters=params,
        )
