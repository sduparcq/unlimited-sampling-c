from .signal import SignalParameters, Signal
import jax.numpy as jnp
import numpy as np


def mod_sample(samples: jnp.ndarray, L: float) -> jnp.ndarray:
    return samples - 2 * L * jnp.floor(samples / (2 * L) + 0.5)


class SignalFactory:
    def __init__(self):
        pass

    def generate_signal(self, signal_params: SignalParameters) -> Signal:
        if signal_params.pt == True:
            signal = self._generate_poly_trig(signal_params)
        else:
            signal = self._generate_pw(signal_params)

        Te = (jnp.pi / signal_params.Omega) * signal_params.M
        support = jnp.arange(signal_params.N) * Te
        mod_samples = mod_sample(samples=signal, L=signal_params.L)

        rng_noise = np.random.default_rng(signal_params.n_seed)

        noise = rng_noise.normal(0, signal_params.sigma, size=signal_params.N)

        return Signal(
            support=support,
            signal=signal,
            mod_samples=mod_samples,
            nms=mod_samples + noise,
            signal_parameters=signal_params,
        )

    def _generate_poly_trig(self, signal_params) -> jnp.ndarray:

        rng = np.random.default_rng(signal_params.s_seed)

        Te = (jnp.pi / signal_params.Omega) * signal_params.M

        omega_grid = 2 * jnp.pi * jnp.fft.fftfreq(signal_params.N, d=Te)

        Z = rng.standard_normal(signal_params.N) + 1j * rng.standard_normal(
            signal_params.N
        )
        H = (jnp.abs(omega_grid) <= signal_params.Omega).astype(float)
        F = Z * H

        f = jnp.real(jnp.fft.ifft(F))
        f = f - f[0]
        f = f / jnp.max(jnp.abs(f))
        return f

    def _generate_pw(self, signal_params: SignalParameters) -> jnp.ndarray:

        Te = (jnp.pi / signal_params.Omega) * signal_params.M

        support = jnp.arange(signal_params.N) * Te

        rng = np.random.default_rng(signal_params.s_seed)

        random_coefs = jnp.asarray(rng.standard_normal(signal_params.N))

        n = jnp.arange(signal_params.N)

        basis = jnp.sinc(
            (signal_params.Omega / jnp.pi)
            * (support[:, None] - n[None, :] * (jnp.pi / signal_params.Omega))
        )
        s = basis @ random_coefs
        s = s - s[0]
        s = s / jnp.max(jnp.abs(s))
        return s
