'''-----------------------------------------------------------
    Image Processing for SAR Signal with Ionospheric Effects
'''

import numpy as np
import copy
from multiprocessing import Pool, cpu_count
from Helper import get_index
from WindowFuncs import *

class Image:
    def __init__(self, domain, window_func, signal, psi_obj, **params):
        """Initialize the Image object with given parameters."""
        self.domain = domain
        self.window_func = window_func
        self.signal = signal
        self.psi = psi_obj
        self.F = params.get('F', 100)
        self.dx = params.get('dx', 0.25)
        self.integrand_cache = {}
        self.image_mat = np.vstack((self.domain, self._evaluate_image()))

    def _evaluate_image(self):
        """Precompute the image integral for all points using the given domain."""
        print("Signal shape:", self.signal.shape)
        real_signal = np.real(self.signal[0, :])
        imag_val = np.empty_like(self.signal[1, :], dtype='complex128')

        for yidx, y in enumerate(self.domain):
            x0 = max(real_signal[0], y - self.F / 2)
            x1 = min(real_signal[-1], y + self.F / 2)
            print(x0, x1)
            mask = (real_signal >= x0) & (real_signal <= x1)
            print(real_signal)
            base = real_signal[mask]
            print(base.shape)
            signal_vals = self.signal[1, mask]
            print(signal_vals.shape)
            print("----")
            waveform = np.exp(-1j * np.pi * (base - y) ** 2 / self.F)
            psi_vals = np.exp(1j * self.psi.calc_psi_cache(y))
            window = self.window_func(base)


            print("waveform.shape:", waveform.shape)
            print("signal_vals.shape:", signal_vals.shape)
            print("window.shape:", window.shape)

            without_psi_heights = waveform * signal_vals * window
            self.integrand_cache[y] = without_psi_heights
            print("without_psi_heights.shaoe:", without_psi_heights.shape)
            print("psi_vals.shape:", psi_vals.shape)
            heights = without_psi_heights * psi_vals

            imag_val[yidx] = np.trapz(heights, base, self.dx) / self.F

        return imag_val

    def update_image(self, psi_func):
        """Update the image by reevaluating with a new psi function."""
        self.psi = psi_func
        for yidx, y in enumerate(self.domain):
            integrand = self.integrand_cache[y] * np.exp(1j * psi_func.calc_psi_cache(y))
            self.image_mat[1, yidx] = np.trapz(integrand, dx=self.dx) / self.F

    def calc_cost(self):
        """Calculate the cost using the fourth power sum of the image matrix values."""
        return np.nansum(np.abs(self.image_mat[1, :]) ** 4)

    def calc_grad(self):
        """Calculate the gradient of the image using precomputed integrands and psi values."""
        image_2norm = 2 * np.abs(self.image_mat[1, :]) ** 2
        Nharmonics = self.psi.Nharmonics
        y_vals = np.size(self.domain)
        partial_mat = np.zeros((2 * Nharmonics, y_vals), dtype='complex128')

        for yidx, y in enumerate(self.domain):
            sarr = self.psi.cache_y2s[y]
            common_integrand_base = self.integrand_cache[y] * np.exp(1j * self.psi.calc_psi_cache(y))

            for idx in range(Nharmonics):
                wavenum_sarr = self.psi.wavenums[idx] * sarr
                cos_integrand = 1j * common_integrand_base * np.cos(wavenum_sarr)
                sin_integrand = -1j * common_integrand_base * np.sin(wavenum_sarr)

                partial_mat[idx, yidx] = np.trapz(cos_integrand, dx=self.dx) / self.F
                partial_mat[Nharmonics + idx, yidx] = np.trapz(sin_integrand, dx=self.dx) / self.F

        prod_rule = np.conjugate(partial_mat) * self.image_mat[1, :] + np.conjugate(self.image_mat[1, :]) * partial_mat
        grad_mat_ampsxN = image_2norm * prod_rule

        return np.nansum(grad_mat_ampsxN, axis=1)
