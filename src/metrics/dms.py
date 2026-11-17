from dataclasses import dataclass
from .common import uot_distance, is_recovered, get_true_measure_from_signal


@dataclass
class DMSMetrics:
    is_recovered: int
    uot_distance_q: float
    threshold_gap: float


def threshold_gap(e, j_true):
    active_support = abs(j_true[1::]) > 0
    e_active = e[active_support]
    e_inactive = e[~active_support]
    if len(e_inactive) == 0 or len(e_active) == 0:
        return -1
    max_inactive = max(e_inactive)
    min_active = min(e_active)
    if min_active > max_inactive:
        return min_active - max_inactive
    return -1


def compute_dms_metrics(signal, result):

    j_rec = result.jumps
    j_true = get_true_measure_from_signal(signal)

    return DMSMetrics(
        is_recovered=is_recovered(j_true, j_rec),
        uot_distance_q=uot_distance(j_true, j_rec),
        threshold_gap=threshold_gap(result.solve_result.e, j_true),
    )
