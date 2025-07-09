
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
from jax import jit
import jax.numpy as jnp
import jax.scipy.integrate as integrate

# Custom imports
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the target directory (`/home/houtlaw/iono-net/model`)
model_dir = os.path.abspath(os.path.join(current_dir, "../../model"))

# Add the model directory to sys.path if it's not already there
if model_dir not in sys.path:
    sys.path.append(model_dir)

print(f"Added {model_dir} to sys.path")
from Helper import *
from Image import *
from Psi import *
from Optimize import *
from model_color import ConfigurableModel

# Matplotlib settings
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (30, 8)
plt.rcParams["savefig.dpi"] = 300

# Parameters
SAMPLE_IDX = 40
DX = 0.25
ISLR_RADIUS = 5 # min distance between scatterers
ISLR_RADIUS_RATIO = 0.6 # ratio of radius for sidelobe integral
ISLR_MAIN_LOBE_WIDTH = 0.75 #fixed main lobe width
COMPARISON_SAMPLE_SIZE = 10

# File paths
DATA_DIR = "/home/houtlaw/iono-net/data/0.5iono"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250625_123749.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250625_123749.json"
SCATTERER_PATH_RELNOISE = f"{DATA_DIR}/test_nuStruct_withSpeckle_20250625_123748.csv"
SIGNAL_PATH_RELNOISE = f"{DATA_DIR}/test_uscStruct_vals_20250625_123749.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250625_123749.csv"
PSI_COEFFS_PATH_RELNOISE = f"{DATA_DIR}/test_compl_ampls_20250625_123749.csv"
MODEL_WEIGHTS_PATH = "/home/houtlaw/iono-net/model/model_weights_full_20250701_094140.pkl"

# Helper Functions
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    normalized = matrix / amp_max
    return normalized.real + 1j * normalized.imag

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def build_psi_values(kpsi, compl_ampls, x):
    val = np.zeros_like(x, dtype=float)
    for ik in range(len(compl_ampls)):
        val += np.real(compl_ampls[ik] * np.exp(1j * kpsi[ik] * x))
    return val

def add_plot_subtitle(setup):
    subtitle = (f"ionoAmplOverPi: {setup['ionoAmplOverPi']}, "
                f"addSpeckleCoeff: {setup['addSpeckleCoeff']}, "
                f"relNoiseCoeff: {setup['relNoiseCoeff']}")
    plt.figtext(0.5, 0.01, subtitle, wrap=True, horizontalalignment='center', fontsize=16)

def compute_islr(image_integral, known_scatterers, x_vals, radius, radius_ratio, main_lobe_width, dx):
    islrs = []
    peaks = [x_vals[i] for i, scatterer in enumerate(known_scatterers) if scatterer > 2]

    image_integral = image_integral ** 2
    total_integral = integrate.trapezoid(image_integral, x=x_vals, dx=dx)

    for peak in peaks:
        inner_indices = [i for i, x in enumerate(x_vals) if np.abs(x - peak) <= main_lobe_width]
        outer_indices = [i for i, x in enumerate(x_vals) if np.abs(x - peak) > main_lobe_width and np.abs(x - peak) <= (radius * radius_ratio)]

        inner_peak_bounds = x_vals[inner_indices]
        outer_peak_bounds = x_vals[outer_indices]

        inner_integral = integrate.trapezoid(image_integral[inner_indices], x=inner_peak_bounds, dx=dx)
        outer_integral = integrate.trapezoid(image_integral[outer_indices], x=outer_peak_bounds, dx=dx)

        islr = 10 * np.log10(outer_integral / inner_integral) if inner_integral != 0 else 0
        islrs.append(islr)

    return np.mean(islrs)

# Load and Process Data
def load_data():
    # Load x_range
    x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values

    # Load setup parameters
    with open(SETUP_PATH) as f:
        setup = json.load(f)

    # Load true scatterers with noise
    true_scatterers = pd.read_csv(SCATTERER_PATH_RELNOISE).map(convert_to_complex).iloc[:, SAMPLE_IDX]
    true_scatterers = true_scatterers.map(np.abs).values

    # Load signal data
    signal_data = pd.read_csv(SIGNAL_PATH_RELNOISE).map(convert_to_complex).T.iloc[SAMPLE_IDX].values
    signal_vals = np.vstack((x_range, signal_data))

    # Load kpsi values
    kpsi_values = pd.read_csv(KPSI_PATH).values

    # Load psi coefficients
    psi_coeffs_df = pd.read_csv(PSI_COEFFS_PATH_RELNOISE).T
    psi_coeffs_df = psi_coeffs_df.map(lambda x: complex(x.replace('i', 'j')))
    psi_coeffs_vals = psi_coeffs_df.iloc[SAMPLE_IDX].values

    return x_range, setup, true_scatterers, signal_vals, kpsi_values, psi_coeffs_vals


def evaluate_image(domain, window_func, signal, cos_coeffs, sin_coeffs, wavenums, F, dx, xi):
    """
    Evaluate the image integral for the given domain, window function, signal, and parameters.
    
    Parameters:
        domain (array): The domain over which to evaluate the image.
        window_func (function): Windowing function applied to the signal.
        signal (ndarray): 2D array where signal[0, :] is the real signal and signal[1, :] is the signal values.
        cos_coeffs (array): Cosine coefficients for the Psi function.
        sin_coeffs (array): Sine coefficients for the Psi function.
        wavenums (array): Array of wavenumbers for the Psi function.
        Nharmonics (int): Number of harmonics for Psi.
        F (float): Parameter for the integration range.
        dx (float): Sampling interval for the domain.
        xi (float): Parameter for the s-arr calculation in Psi.
    
    Returns:
        ndarray: The evaluated image integral values.
    """
    real_signal = np.real(signal[0, :])
    imag_val = np.empty_like(signal[1, :], dtype='complex128')

    def calc_sarr_linear(xarr, y, xi):
        """Calculate s-arr values linearly."""
        return (xarr - y) / xi

    def calc_psi_cache(base, y, cos_coeffs, sin_coeffs, wavenums, xi):
        """Cache psi values for a given y."""
        sarr =  xi * base + (1 - xi) * y  # Calculate sarr directly using the base array
        psi_vals = np.real(np.sum(
            cos_coeffs * np.cos(np.outer(sarr, wavenums)) -
            sin_coeffs * np.sin(np.outer(sarr, wavenums)),
            axis=1
        ))
        return psi_vals

    for yidx, y in enumerate(domain):
        x0 = max(real_signal[0], y - F / 2)
        x1 = min(real_signal[-1], y + F / 2)
        mask = (real_signal >= x0) & (real_signal <= x1)

        base = real_signal[mask]
        signal_vals = signal[1, mask]
        waveform = np.exp(-1j * np.pi * (base - y) ** 2 / F)
        psi_vals = np.exp(1j * calc_psi_cache(base, y, cos_coeffs, sin_coeffs, wavenums, xi))
        window = window_func(base)

        heights = waveform * signal_vals * window * psi_vals
        imag_val[yidx] = np.trapz(heights, base, dx) / F

    return imag_val

def main():
    # Load data
    x_range, setup, true_scatterers, signal_vals, kpsi_values, psi_coeffs_vals = load_data()

    # === NEW PART: Trim leading/trailing zero-padding from signal ===
    # If signal is complex, use magnitude for detecting zeros
    signal_array = signal_vals[1, :]  # the 2nd row are the signal samples
    nonzero_mask = np.abs(signal_array) > 1e-12
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        # Edge case: everything is zero; no trimming possible
        print("All signal samples are zero. Using the original arrays.")
    else:
        start_idx, end_idx = nonzero_indices[0], nonzero_indices[-1]
        
        # Trim x_range and signal
        trimmed_x = x_range[start_idx : end_idx + 1]
        trimmed_signal = signal_array[start_idx : end_idx + 1]
        
        # Reassign for use in the rest of the pipeline
        x_range_trunc = trimmed_x
        signal_vals_trunc = np.vstack((trimmed_x, trimmed_signal))
        print(f"Trimmed signal shape: {signal_vals_trunc.shape}, new x-range shape: {x_range_trunc.shape}")

    # Retrieve parameters
    F, ionoNHarm, xi, windowType, sumType = (
        setup["F"], 
        setup["ionoNharm"], 
        setup["xi"], 
        setup["windowType"], 
        setup["sumType"],
    )

    # Split psi coefficients into cosine and sine components
    cos_coeffs = [j.real for j in psi_coeffs_vals]
    sin_coeffs = [-j.imag for j in psi_coeffs_vals]

    # Create Psi object
    rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
    rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)

    # Create image object
    image_object = Image(x_range_trunc, window_func=rect_window, signal=signal_vals_trunc, psi_obj=rec_fourier_psi, F=F)
    image_integral = image_object._evaluate_image()

    zero_psi = RecFourierPsi(np.zeros_like(cos_coeffs), np.zeros_like(sin_coeffs), kpsi_values, ionoNHarm)
    zero_psi.cache_psi(x_range_trunc, F, DX, xi)
    unfocused_image_object = Image(x_range_trunc, window_func = rect_window, signal = signal_vals_trunc, psi_obj = zero_psi, F = F)

    # Plot results
    plt.plot(x_range, true_scatterers, 'orange', lw=3)
    plt.plot(x_range_trunc, np.abs(image_integral) / DX, lw=2)
    plt.title("Image Integral")
    add_plot_subtitle(setup)
    plt.legend(["True Point Scatterers", "Image Integral"])
    plt.savefig("image_integral.png")

    

    # Load model weights
    with open(MODEL_WEIGHTS_PATH, 'rb') as f:
        params = pickle.load(f)

    # Define model
    architecture =  [256,256,128,128,64,64,32,32] 
    model = ConfigurableModel(architecture=architecture, activation_fn=jax.numpy.tanh)

    # Run inference on the trimmed signal
    signal_df_full = pd.read_csv(SIGNAL_PATH_RELNOISE, dtype=str)
    complex_matrix = signal_df_full.map(convert_to_complex).to_numpy().T  # shape (num_samples, signal_len)
    normalized_matrix = normalize_complex_to_unit_range(complex_matrix)
    fft_matrix = normalized_matrix # DISABLED FFT FOR NOW
    sample_fft = fft_matrix[SAMPLE_IDX]
    
    model_input = jnp.expand_dims(split_complex_to_imaginary(sample_fft), axis=0)
    
    #model_input = split_complex_to_imaginary(sample_fft)
    model_output = model.apply({'params': params}, model_input, deterministic=True)
    model_output = model_output[0]  # remove batch dim: shape (12,)
    half = model_output.shape[0] // 2
    model_output_complex = model_output[:half] + 1j * model_output[half:]

    # Split model output into cosine and sine coefficients
    model_cos_coeffs = [j.real for j in model_output_complex]
    model_sin_coeffs = [-j.imag for j in model_output_complex]

    # Create model Psi object
    model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
    model_rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)

    # Create model image object
    model_image_object = Image(x_range_trunc, window_func=rect_window, signal=signal_vals_trunc, psi_obj=model_rec_fourier_psi, F=F)
    model_image_integral = model_image_object._evaluate_image()
    zero_focus_integral = unfocused_image_object._evaluate_image()


    # Plot model results
    plt.figure()
    plt.plot(x_range, true_scatterers, 'orange', lw=3)
    plt.plot(x_range_trunc, np.abs(model_image_integral)/DX, lw=2)
    plt.plot(x_range_trunc, np.abs(zero_focus_integral)/DX, lw = 2)
    plt.title("Image Integral (NN) Inference, ionoAmplOverPi = 0.5")
    add_plot_subtitle(setup)
    plt.legend(["True Point Scatterers", "Image Integral (NN)", "Image Integral (No Focus)"])
    plt.savefig("neural_image_integral.png")

    # Compute Psi Values and Plot
    model_psi_vals = build_psi_values(kpsi_values, model_output_complex, x_range_trunc)
    true_psi_vals = build_psi_values(kpsi_values, psi_coeffs_vals, x_range_trunc)
    print("MODEL ERROR", model_output_complex - psi_coeffs_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range_trunc, true_psi_vals, label="True Psi Values", lw=4)
    plt.plot(x_range_trunc, model_psi_vals, label="Network Psi Values")
    plt.xlabel("x")
    plt.ylabel("Psi")
    plt.title("Psi Values Plot")
    add_plot_subtitle(setup)
    plt.grid(True)
    plt.legend()
    plt.savefig('psi_comparison.png')

    # Compute ISLR
    islr_avg_nn = compute_islr(np.abs(model_image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
    print("ISLR Average (NN):", islr_avg_nn)

    islr_avg_grad = compute_islr(np.abs(image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
    print("ISLR Average (Classic):", islr_avg_grad)

if __name__ == "__main__":
    main()