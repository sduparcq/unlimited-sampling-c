from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class SignalParameters:
    N: int
    L: float
    Omega: float
    bounds: tuple
    M: float
    sigma: float
    pt: bool
    s_seed: int
    n_seed: int


@dataclass
class Signal:
    support: jnp.ndarray
    signal: jnp.ndarray
    mod_samples: jnp.ndarray
    nms: jnp.ndarray  # nms = noisy mod samples -> too long
    signal_parameters: SignalParameters
