from dataclasses import dataclass

from ..methods import *

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


@dataclass
class GLExperiment:
    signal: SignalParameters
    preparation: GLPreparationParameters
    solver: GLParameters
    post_process: GLPostProcessParameters
