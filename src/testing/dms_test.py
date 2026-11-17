# %% Imports

import matplotlib.pyplot as plt
import jax.numpy as jnp

from src.methods.dms import (
    DMS,
    DMSParameters,
    DMSPostProcessParameters,
    DMSPreparationParameters,
)
from src.signals.factory import SignalFactory
from src.signals.signal import SignalParameters
from src.metrics.common import get_true_measure_from_signal


# %% Signal generation

M = 0.05
Omega = 10.0
bounds = (0.0, 1.0)

Te = (jnp.pi / Omega) * M
N = int(jnp.ceil(1.0 + (bounds[1] - bounds[0]) / Te))

signal_params = SignalParameters(
    N=N,
    M=M,
    Omega=Omega,
    L=0.3,
    sigma=0.09,
    bounds=bounds,
    s_seed=1,
    n_seed=2,
    pt=False,
)

factory = SignalFactory()
signal = factory.generate_signal(signal_params)


# %% DMS preparation and solving

dms = DMS()
prepared = dms.prepare(params=DMSPreparationParameters(), N=N)

solve_params = DMSParameters(
    ext_iter=100,
    int_iter=100,
    alpha=0.015,
    beta=0.18,
)

solve_result = prepared.solve(
    nms=signal.nms,
    params=solve_params,
)

# %% Threshold gap


def threshold_gap(e, j_true):
    active_support = abs(j_true[1::]) > 0
    e_active = e[active_support]
    max_inactive = max(e[~active_support])
    min_active = min(e_active)

    if min_active > max_inactive:
        return max_inactive, min_active
    return 0, 0


j_true = get_true_measure_from_signal(signal)
min_val, max_val = threshold_gap(solve_result.e, j_true)

# %% Post-processing

post_process_params = DMSPostProcessParameters(
    L=0.3,
    threshold=0.5,
)

result = dms.post_process(
    nms=signal.nms,
    solve_result=solve_result,
    params=post_process_params,
)


# %% Plot DMS solution
plt.subplot(2, 1, 1)
plt.plot(signal.signal - signal.mod_samples, label=r"$\varepsilon$")
plt.plot(solve_result.x, label="x")
plt.plot(signal.nms, ls="--", label="nms")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)

plt.plot(solve_result.e, label="e")
plt.xlabel("support")
plt.ylabel("e")

plt.axhline(min_val, color="red", ls="--", label="min threshold")
plt.axhline(max_val, color="red", ls="--", label="max threshold")

plt.axhspan(max_val, min_val, color="green", alpha=0.2, label="Valid thresholds")
plt.axhspan(0, max_val, color="gray", alpha=0.15)
plt.axhspan(min_val, plt.ylim()[1], color="gray", alpha=0.15)

plt.legend()
plt.show()


# %% Plot reconstruction

t = signal.support

plt.figure(figsize=(10, 5))

plt.plot(t, signal.signal, label="Signal")
plt.plot(t, signal.mod_samples, label="Modulo samples")
plt.plot(
    t,
    result.recovered_signal,
    "--",
    label="DMS reconstruction",
)

plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("DMS reconstruction")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# %% Diagnostics

print("N =", N)
print("Te =", Te)
print("D shape =", prepared.D.shape)

print("x finite =", jnp.all(jnp.isfinite(solve_result.x)))
print("e finite =", jnp.all(jnp.isfinite(solve_result.e)))
print(
    "recovered finite =",
    jnp.all(jnp.isfinite(result.recovered_signal)),
)
print("jumps =", result.jumps)
