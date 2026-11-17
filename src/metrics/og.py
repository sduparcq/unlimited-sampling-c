from dataclasses import dataclass
from .common import uot_distance, is_recovered, get_true_measure_from_signal


@dataclass
class OGMetrics:
    is_recovered: int
    uot_distance_q: float


def compute_og_metrics(signal, result):

    j_rec = result.jumps
    j_true = get_true_measure_from_signal(signal)

    return OGMetrics(
        is_recovered=is_recovered(j_true, j_rec),
        uot_distance_q=uot_distance(j_true, j_rec),
    )
