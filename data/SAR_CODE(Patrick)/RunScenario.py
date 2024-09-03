'''
Build Scenario and Create SAR Image

This script outlines the procedure to simulate a SAR image including target signal creation with clutter,
applying noise, and ionospheric phase perturbation via Fourier series.
'''

import numpy as np
from Helper import build_Pure_Target, add_Clutter, evaluate_Signal_Integral, add_Noise
from Psi import RecFourierPsi
from Image import Image
from WindowFuncs import rect_window, parab_scaled_window
from Optimize import local

# Define the simulation parameters
target_span = (0, 360)
dx = 0.25
F = 100
xi = 0.5
target_dict = {144: 1, 186: 1, 210: 1}
target_clutter = 0.1
target_clutter_seed = 13
signal_noise = 0.05
signal_noise_seed = 78
zeta = 0.6  # Regularization factor

# Initialize target signals with clutter
pure_target = build_Pure_Target(target_span, target_dict, dx)
target = add_Clutter(pure_target, target_clutter, dx=dx, seed=target_clutter_seed)

# Setup the domain for PSI function
domain = np.real(target[0, :])

# Initialize the Fourier Psi function with coefficients
cos_amps = np.array([-0.8136, -1.2131, 0.6413, 0.2349, 0.0852, -0.1096])
sin_amps = np.array([5.9878, 0.9003, 0.1987, -0.2957, 0.2262, -0.1271])
wavenumbers = np.array([0.0377, 0.0754, 0.1131, 0.1508, 0.1885, 0.2262])
harmonics = len(wavenumbers)

# Create a RecFourierPsi object and cache the Psi values
psi_func = RecFourierPsi(cos_amps, sin_amps, wavenumbers, harmonics)
psi_func.cache_psi(domain, F, dx, xi)

# Compute the signal integral with Psi effect
pure_signal = evaluate_Signal_Integral(target, psi_func, F, dx)
signal = add_Noise(pure_signal, signal_noise, seed=signal_noise_seed)
signal[1, :] /= dx  # Normalize the signal amplitude

# Zero amplitude Psi function for baseline comparison
cos_zero, sin_zero = np.zeros_like(cos_amps), np.zeros_like(sin_amps)
psi_zero = RecFourierPsi(cos_zero, sin_zero, wavenumbers, harmonics)
psi_zero.cache_psi(domain, F, dx, xi)

# Determine the window function for imaging
image_window = parab_scaled_window if 'scaled' in 'parab_scaled_window' else rect_window

# Create image objects with zero and actual Psi perturbations
image_zero_psi = Image(domain, image_window, signal, psi_zero, F=F, dx=dx)
image = Image(domain, image_window, signal, psi_func, F=F, dx=dx)

# Perform optimization to refine Psi amplitudes
amps_vec = np.concatenate((cos_amps, sin_amps))
optim_results = local(amps_vec, psi_zero, image_zero_psi, zeta)

# Update Psi and image with optimized parameters
N = harmonics
psi_zero.update_amps(optim_results['x'][:N], optim_results['x'][N:2 * N])
image_zero_psi.update_image(psi_zero)  # Autofocused Image

# Output the results
print("Optimization completed. Optimized image updated.")
