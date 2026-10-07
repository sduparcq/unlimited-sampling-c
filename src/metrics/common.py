import numpy as np
import ot
import matplotlib.pyplot as plt


def get_true_measure_from_signal(signal):
    return np.diff(signal.signal - signal.mod_samples, prepend=0)


def uot_distance(j_true, j_rec) -> float:
    tol = 1e-12
    reg = 0.01
    reg_M = 0.1

    N = len(j_true)

    x_grid = np.linspace(0, 1, N)

    def extract_sparse_measure(v):
        idx = np.where(np.abs(v) > tol)[0]
        return x_grid[idx], v[idx]

    x_true, a_true = extract_sparse_measure(j_true)
    x_rec, a_rec = extract_sparse_measure(j_rec)

    def split_signed_measure(x, a):
        pos = a > tol
        neg = a < -tol

        return (
            x[pos],
            a[pos],
            x[neg],
            -a[neg],
        )

    x_true_p, a_true_p, x_true_m, a_true_m = split_signed_measure(
        x_true,
        a_true,
    )

    x_rec_p, a_rec_p, x_rec_m, a_rec_m = split_signed_measure(
        x_rec,
        a_rec,
    )

    def solve_uot_component(x1, a1, x2, a2):
        if len(a1) == 0 and len(a2) == 0:
            return 0.0

        if len(a1) == 0:
            return float(reg_M * np.sum(a2))

        if len(a2) == 0:
            return float(reg_M * np.sum(a1))

        M12 = ot.dist(
            x1.reshape(-1, 1),
            x2.reshape(-1, 1),
            metric="sqeuclidean",
        )

        M11 = ot.dist(
            x1.reshape(-1, 1),
            x1.reshape(-1, 1),
            metric="sqeuclidean",
        )

        M22 = ot.dist(
            x2.reshape(-1, 1),
            x2.reshape(-1, 1),
            metric="sqeuclidean",
        )

        def uot_cost(a, b, M):
            return float(
                ot.unbalanced.sinkhorn_unbalanced2(
                    np.asarray(a),
                    np.asarray(b),
                    np.asarray(M),
                    reg,
                    reg_M,
                )
            )

        C12 = uot_cost(a1, a2, M12)
        C11 = uot_cost(a1, a1, M11)
        C22 = uot_cost(a2, a2, M22)

        return C12 - 0.5 * C11 - 0.5 * C22

    score_positive = solve_uot_component(
        x_true_p,
        a_true_p,
        x_rec_p,
        a_rec_p,
    )

    score_negative = solve_uot_component(
        x_true_m,
        a_true_m,
        x_rec_m,
        a_rec_m,
    )

    return score_positive + score_negative


def is_recovered(j_true, j_rec) -> int:
    # plt.plot(j_rec, label="j rec")
    #
    # plt.plot(j_true, label="j true", linestyle="--")
    # plt.show()
    #
    if np.sum(np.abs(j_true - j_rec)) <= 0.001:
        return 1
    return 0
