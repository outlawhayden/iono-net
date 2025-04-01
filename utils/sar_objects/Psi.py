import jax.numpy as jnp
from jax import vmap
from functools import partial

''' ------------------------------------------------------------------
    Base Psi Class for Ionospheric Phase Perturbation (JAX)
'''
class Psi:
    def __init__(self, domain=jnp.arange(0, 360 + 1/8, 1/8), xi=0.5):
        self.domain = domain
        self.xi = xi
        self.psi_cache = {}

    def calc_sarr_linear(self, x, yORz):
        return self.xi * x + (1 - self.xi) * yORz

''' ------------------------------------------------------------------
    Reconstructed Fourier Perturbation Function (JAX)
'''
class RecFourierPsi(Psi):
    def __init__(self, cosAmps, sinAmps, wavenums, Nharmonics):
        super().__init__()
        self.cosAmps = cosAmps
        self.sinAmps = sinAmps
        self.wavenums = wavenums
        self.Nharmonics = Nharmonics
        self.vertShift = 0
        self.cache_y2s = {}
        self.cache_y2psi = {}

    def cache_psi(self, domain, F, dx, xi):
        for i, y in enumerate(domain):
            xarr = domain[jnp.maximum(0, i - int(F/2/dx)): i + int(F/2/dx) + 1]
            sarr = self.calc_sarr_linear(xarr, y)
            self.cache_y2s[y.item()] = sarr
            psi_vals = self.calc_psi(sarr)
            self.cache_y2psi[y.item()] = psi_vals

    def calc_psi(self, sarr):
        cos_terms = jnp.cos(jnp.outer(sarr, self.wavenums)) * self.cosAmps
        sin_terms = jnp.sin(jnp.outer(sarr, self.wavenums)) * self.sinAmps
        return jnp.sum(cos_terms + sin_terms, axis=1)

    def calc_psi_cache(self, y):
        return self.cache_y2psi[y.item()]

    def update_amps(self, cosAmps, sinAmps):
        self.cosAmps = cosAmps
        self.sinAmps = sinAmps
        for y, s_val in self.cache_y2s.items():
            self.cache_y2psi[y] = self.calc_psi(s_val)