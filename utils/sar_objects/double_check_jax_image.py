# Imports and Setup
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import jax
import jax.numpy as jnp
import jax.scipy.integrate as integrate

jax.config.update("jax_enable_x64", True)

# Custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "../../model"))
if model_dir not in sys.path:
    sys.path.append(model_dir)
print(f"Added {model_dir} to sys.path")

from Helper import *
from Image import *
from Psi import *
from Optimize import *
from model import ConfigurableModel

# Matplotlib settings
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (30, 8)
plt.rcParams["savefig.dpi"] = 300

# Parameters
SAMPLE_IDX = 2
DX = 0.25
zero_pad = 50
ISLR_RADIUS = 5
ISLR_RADIUS_RATIO = 0.6
ISLR_MAIN_LOBE_WIDTH = 0.75

# Paths
DATA_DIR = "/home/houtlaw/iono-net/data/baselines/10k_lownoise"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250206_104914.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250206_104914.json"
SCATTERER_PATH_RELNOISE = f"{DATA_DIR}/test_nuStruct_withSpeckle_20250206_104911.csv"
SIGNAL_PATH_RELNOISE = f"{DATA_DIR}/test_uscStruct_vals_20250206_104913.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250206_104914.csv"
PSI_COEFFS_PATH_RELNOISE = f"{DATA_DIR}/test_compl_ampls_20250206_104913.csv"

# Helper Functions
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace('i', 'j'))

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def load_data():
    x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values
    with open(SETUP_PATH) as f:
        setup = json.load(f)
    signal_data = pd.read_csv(SIGNAL_PATH_RELNOISE).map(convert_to_complex).T.iloc[SAMPLE_IDX].values
    kpsi_values = pd.read_csv(KPSI_PATH).values.flatten()
    psi_coeffs_df = pd.read_csv(PSI_COEFFS_PATH_RELNOISE).T
    psi_coeffs_vals = psi_coeffs_df.map(lambda x: complex(x.replace('i', 'j'))).iloc[SAMPLE_IDX].values
    return x_range, setup, signal_data, kpsi_values, psi_coeffs_vals

def jnp_image_reconstruction(x_range, signal_vals, pr, pi, kpsi_values, F, dx, xi):
    # Signal is only defined when away by F/2 from boundaries
    F2 = F / 2

    # Find x_range_trimmed (keep only where x is in [F2, max(x) - F2])
    mask = (x_range >= (x_range[0] + F2)) & (x_range <= (x_range[-1] - F2))
    x_trimmed = x_range[mask]
    signal_trimmed = signal_vals[mask]

    # Define offsets based on F
    offsets = jnp.linspace(-F2, F2, int(F/dx)+1)

    def trapz_nonuniform_jax(y, x):
        dx = x[1:] - x[:-1]
        avg_y = 0.5 * (y[1:] + y[:-1])
        return jnp.sum(dx * avg_y)

    def eval_point(y):
        base = y + offsets
        interp = jnp.interp(base, x_trimmed, signal_trimmed)
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_pred = jnp.exp(1j * (jnp.sum(pr[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0) +
                                jnp.sum(pi[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)))
        integrand = waveform * interp * psi_pred
        return trapz_nonuniform_jax(integrand, base) / F


    image_reconstruction = jax.vmap(eval_point)(x_trimmed)
    return x_trimmed, image_reconstruction

def main():
    # Load data
    x_range, setup, signal_data, kpsi_values, psi_coeffs_vals = load_data()
    F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]

    F2 = F / 2

    # === Trim Signal for External Library ===
    mask_signal = (x_range >= (x_range[0] + F2)) & (x_range <= (x_range[-1] - F2))
    x_range_signal = x_range[mask_signal]
    signal_vals_signal = signal_data[mask_signal]

    # External Fourier Psi Reconstruction (signal domain)
    cos_coeffs = [j.real for j in psi_coeffs_vals]
    sin_coeffs = [-j.imag for j in psi_coeffs_vals]
    rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
    rec_fourier_psi.cache_psi(x_range_signal, F, DX, xi)
    signal_stacked = np.vstack((x_range_signal, signal_vals_signal))
    image_object = Image(x_range_signal, window_func=rect_window, signal=signal_stacked, psi_obj=rec_fourier_psi, F=F)
    image_integral_external = image_object._evaluate_image()

    # === JAX Image Reconstruction ===
    pr = jnp.array([j.real for j in psi_coeffs_vals])
    pi = jnp.array([-j.imag for j in psi_coeffs_vals])
    x_recon_signal, image_integral_jnp_signal = jnp_image_reconstruction(x_range, signal_data, pr, pi, kpsi_values, F, DX, xi)

    mask_image = (x_recon_signal >= (x_recon_signal[0] + F/2)) & (x_recon_signal <= (x_recon_signal[-1] - F/2))
    x_recon = x_recon_signal[mask_image]
    image_integral_jnp = image_integral_jnp_signal[mask_image]

    # Also trim external image_integral_external
    image_integral_external = image_integral_external[mask_image]

    # === Compute Magnitude and Phase Differences ===
    mag_external = np.abs(image_integral_external) / DX
    mag_jax = np.abs(image_integral_jnp) / DX
    magnitude_difference = mag_external - np.array(mag_jax)

    phase_external = np.angle(image_integral_external)
    phase_jax = jnp.angle(image_integral_jnp)
    phase_difference = phase_external - np.array(phase_jax)
    phase_difference_wrapped = (phase_difference + np.pi) % (2 * np.pi) - np.pi

    # === Plot Magnitude difference
    plt.figure(figsize=(12,6))
    plt.plot(x_recon, magnitude_difference, label='Magnitude Difference (External - JAX)', linewidth=3)
    plt.title('Magnitude Difference Between Reconstructions')
    plt.xlabel('x')
    plt.ylabel('Magnitude Difference')
    plt.legend()
    plt.grid(True)
    plt.savefig("compare_image_integrals_magnitude_difference.png")
    plt.show()

    # === Plot Phase difference
    plt.figure(figsize=(12,6))
    plt.plot(x_recon, phase_difference_wrapped, label='Phase Difference (External - JAX)', linewidth=3)
    plt.title('Phase Difference Between Reconstructions')
    plt.xlabel('x')
    plt.ylabel('Phase Difference (radians)')
    plt.legend()
    plt.grid(True)
    plt.savefig("compare_image_integrals_phase_difference.png")
    plt.show()

if __name__ == "__main__":
    main()