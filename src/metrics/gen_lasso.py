from dataclasses import dataclass
from .common import uot_distance, is_recovered, get_true_measure_from_signal


@dataclass
class GLMetrics:
    is_recovered: int
    uot_distance: float
    uot_distance_q: float


def compute_gen_lasso_metrics(signal, result):

    j_rec = result.solve_result.jumps
    j_rec_q = result.jumps
    j_true = get_true_measure_from_signal(signal)
    # check if j_rec_q != j_rec
    return GLMetrics(
        is_recovered=is_recovered(j_true, j_rec_q),
        uot_distance=uot_distance(j_true, j_rec),
        uot_distance_q=uot_distance(j_true, j_rec_q),
    )
