from dataclasses import dataclass

from ..methods.dms import (
    DMSParameters,
    DMSPostProcessParameters,
    DMSPreparationParameters,
)
from ..methods.fista import (
    FistaParameters,
    FistaPostProcessParameters,
    FistaPreparationParameters,
)
from ..signals.signal import SignalParameters


@dataclass
class FistaExperiment:
    signal: SignalParameters
    preparation: FistaPreparationParameters
    solver: FistaParameters
    post_process: FistaPostProcessParameters


@dataclass
class DMSExperiment:
    signal: SignalParameters
    preparation: DMSPreparationParameters
    solver: DMSParameters
    post_process: DMSPostProcessParameters
