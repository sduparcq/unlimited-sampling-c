from dataclasses import dataclass
from .common import uot_distance, is_recovered, get_true_measure_from_signal


@dataclass
class FistaMetrics:
    is_recovered: int
    uot_distance: float
    uot_distance_q: float
    convergence_witness: float


def compute_fista_metrics(signal, result):

    j_rec = result.solve_result.jumps
    j_rec_q = result.jumps
    j_true = get_true_measure_from_signal(signal)

    return FistaMetrics(
        is_recovered=is_recovered(j_true, j_rec_q),
        uot_distance=uot_distance(j_true, j_rec),
        uot_distance_q=uot_distance(j_true, j_rec_q),
        convergence_witness=result.solve_result.convergence_witness,
    )
