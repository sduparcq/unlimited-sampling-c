import matplotlib.pyplot as plt
import jax.numpy as jnp

from ..signals.factory import SignalFactory
from ..signals.signal import SignalParameters


params = SignalParameters(
    N=512,
    M=0.05,
    Omega=10.0,
    L=0.3,
    sigma=0.05,
    bounds=(0, 1),
    s_seed=1,
    n_seed=2,
    pt=True,
)

factory = SignalFactory()
signal = factory.generate_signal(params)

x = signal.support
y = signal.signal

Te = (jnp.pi / params.Omega) * params.M

Y = jnp.fft.fft(y)
freq = 2 * jnp.pi * jnp.fft.fftfreq(params.N, d=Te)

Y_shifted = jnp.fft.fftshift(Y)
freq_shifted = jnp.fft.fftshift(freq)

nyquist = jnp.pi / Te

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(x, y)
axes[0].set_title("Original signal")
axes[0].set_xlabel("t")
axes[0].set_ylabel("f(t)")

axes[1].plot(x, signal.mod_samples)
axes[1].set_title("Modulo samples")
axes[1].set_xlabel("t")
axes[1].set_ylabel("y(t)")

axes[2].plot(freq_shifted, jnp.abs(Y_shifted))
axes[2].axvline(params.Omega, linestyle="--")
axes[2].axvline(-params.Omega, linestyle="--")
axes[2].axvline(nyquist, linestyle=":")
axes[2].axvline(-nyquist, linestyle=":")
axes[2].set_xlim(-nyquist, nyquist)
axes[2].set_title("DFT")
axes[2].set_xlabel("Angular frequency")
axes[2].set_ylabel("|F|")

plt.tight_layout()
plt.show()
