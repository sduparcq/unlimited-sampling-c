# %% Imports

import matplotlib.pyplot as plt
import jax.numpy as jnp

from src.methods.og import (
    OG,
    OGParameters,
    OGPostProcessParameters,
    OGPreparationParameters,
)
from src.signals.factory import SignalFactory
from src.signals.signal import SignalParameters
from src.metrics.common import get_true_measure_from_signal


# %% Signal generation

M = 0.05
Omega = 10.0
bounds = (0.0, 10.0)

Te = (jnp.pi / Omega) * M
N = int(jnp.ceil(1.0 + (bounds[1] - bounds[0]) / Te))

signal_params = SignalParameters(
    N=N,
    M=M,
    Omega=Omega,
    L=0.3,
    sigma=0.01,
    bounds=bounds,
    s_seed=1,
    n_seed=2,
    pt=False,
)

factory = SignalFactory()
signal = factory.generate_signal(signal_params)


# %% OG preparation and solving

dms = OG()
Np = 1
prepared = dms.prepare(params=OGPreparationParameters(Np=Np))

solve_params = OGParameters(Np=Np)

solve_result = prepared.solve(
    signal=signal,
    params=solve_params,
)


# %% Post-processing

post_process_params = OGPostProcessParameters()

result = dms.post_process(
    nms=signal.nms,
    solve_result=solve_result,
    params=post_process_params,
)


# %% Plot reconstruction

t = signal.support

plt.plot(t, result.eps, label=r"$\varepsilon$")
plt.plot(t, result.jumps, label=r"$j$")
plt.legend()
plt.show()

plt.plot(t, result.recovered_signal, label="recovered signal")
plt.plot(t, signal.signal, label="true signal")
plt.legend()
plt.show()
