'''-----------------------------------------------------------
    Image Processing for SAR Signal with Ionospheric Effects (JAX)
'''

import jax.numpy as jnp
from jax import lax
from functools import partial
from jax import vmap
class Image:
    def __init__(self, domain, window_func, signal, psi_obj, **params):
        self.domain = domain
        self.window_func = window_func
        self.signal = signal
        self.psi = psi_obj
        self.F = params.get('F', 100)
        self.dx = params.get('dx', 0.25)
        self.window_size = int(self.F / self.dx) + 1
        self.offsets = jnp.linspace(-self.F / 2, self.F / 2, self.window_size)
        self.image_mat = jnp.stack([self.domain, self._evaluate_image()])

    def _evaluate_image(self):
        real_signal = self.signal[0, :]
        signal_vals = self.signal[1, :]

        def compute_single(y):
            base = y + self.offsets

            real_interp = jnp.interp(base, real_signal, jnp.real(signal_vals))
            imag_interp = jnp.interp(base, real_signal, jnp.imag(signal_vals))
            signal_interp = real_interp + 1j * imag_interp

            window_real = self.window_func(real_signal)
            window_interp = jnp.interp(base, real_signal, window_real)

            waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / self.F)

            sarr = self.psi.calc_sarr_linear(base, y)
            psi_core = jnp.sum(
                self.psi.cosAmps * jnp.cos(jnp.outer(sarr, self.psi.wavenums)) +
                self.psi.sinAmps * jnp.sin(jnp.outer(sarr, self.psi.wavenums)),
                axis=1
            )
            psi_vals = jnp.exp(1j * psi_core)
            integrand = waveform * signal_interp * window_interp * psi_vals

            real_part = jnp.trapz(jnp.real(integrand), dx=self.dx)
            imag_part = jnp.trapz(jnp.imag(integrand), dx=self.dx)
            return (real_part + 1j * imag_part) / self.F

        return vmap(compute_single)(self.domain)

    def calc_cost(self):
        return jnp.sum(jnp.abs(self.image_mat[1, :]) ** 4)