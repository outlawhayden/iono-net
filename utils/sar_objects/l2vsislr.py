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

print("Jax enabled on:", jax.devices())

# Custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "../../model"))

if model_dir not in sys.path:
    sys.path.append(model_dir)

print(f"Added {model_dir} to sys.path")
from Helper import *
from Image import *
from Psi import *
from model import ConfigurableModel

# Matplotlib settings
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (30, 8)
plt.rcParams["savefig.dpi"] = 300

# Parameters
SAMPLE_IDX = 30
TRIM_SIZE = 5
DX = 0.25
ISLR_RADIUS = 5 # min distance between scatterers
ISLR_RADIUS_RATIO = 0.5 # percentage of radius for sidelobe integral
ISLR_MAIN_LOBE_WIDTH = 0.75 #fixed main lobe width
NUM_COMPARISON_SAMPLES = 300

# File paths
DATA_DIR = "/home/houtlaw/iono-net/data/perturbation_experiments/baseline"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250107_122955.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250107_122955.json"
SCATTERER_PATH_RELNOISE = f"{DATA_DIR}/nuStruct_withSpeckle_20250107_122953.csv"
SIGNAL_PATH_RELNOISE = f"{DATA_DIR}/uscStruct_vals_20250107_122955.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250107_122955.csv"
PSI_COEFFS_PATH_RELNOISE = f"{DATA_DIR}/compl_ampls_20250107_122955.csv"
MODEL_WEIGHTS_PATH = f"{DATA_DIR}/model_weights_20250107_130653.pkl"

# Helper Functions
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def build_psi_values(kpsi, compl_ampls, x):
    val = np.zeros_like(x, dtype=float)
    for ik in range(len(compl_ampls)):
        val += np.real(compl_ampls[ik] * np.exp(1j * kpsi[ik] * x))
    return val

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
def load_data(sample_idx):
    x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values

    with open(SETUP_PATH) as f:
        setup = json.load(f)

    true_scatterers = pd.read_csv(SCATTERER_PATH_RELNOISE).map(convert_to_complex).iloc[:, sample_idx]
    true_scatterers = true_scatterers.map(np.abs).values

    signal_data = pd.read_csv(SIGNAL_PATH_RELNOISE).map(convert_to_complex).T.iloc[sample_idx].values
    signal_vals = np.vstack((x_range, signal_data))

    kpsi_values = pd.read_csv(KPSI_PATH).values

    psi_coeffs_df = pd.read_csv(PSI_COEFFS_PATH_RELNOISE).T
    psi_coeffs_df = psi_coeffs_df.map(lambda x: complex(x.replace('i', 'j')))
    psi_coeffs_vals = psi_coeffs_df.iloc[sample_idx].values

    return x_range, setup, true_scatterers, signal_vals, kpsi_values, psi_coeffs_vals


def calculate_islr(model, params, sample_idx):
    x_range, setup, true_scatterers, signal_vals, kpsi_values, psi_coeffs_vals = load_data(sample_idx)

    signal_array = signal_vals[1, :]
    nonzero_mask = np.abs(signal_array) > 1e-12
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        x_range_trunc = x_range
        signal_vals_trunc = signal_vals
    else:
        start_idx, end_idx = nonzero_indices[0], nonzero_indices[-1]
        x_range_trunc = x_range[start_idx:end_idx + 1]
        signal_vals_trunc = np.vstack((x_range_trunc, signal_array[start_idx:end_idx + 1]))

    F, ionoNHarm, xi, windowType, sumType = (
        setup["F"], 
        setup["ionoNharm"], 
        setup["xi"], 
        setup["windowType"], 
        setup["sumType"],
    )

    cos_coeffs = [j.real for j in psi_coeffs_vals]
    sin_coeffs = [-j.imag for j in psi_coeffs_vals]

    rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
    rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)

    image_object = Image(x_range_trunc, window_func=rect_window, signal=signal_vals_trunc, psi_obj=rec_fourier_psi, F=F)
    image_integral = image_object._evaluate_image()

    model_input = split_complex_to_imaginary(signal_vals[1])
    model_output = model.apply({'params': params}, model_input, deterministic=True)
    model_output_complex = model_output[: len(model_output)//2] + 1j * model_output[len(model_output)//2:]

    model_cos_coeffs = [j.real for j in model_output_complex]
    model_sin_coeffs = [-j.imag for j in model_output_complex]

    model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
    model_rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)

    model_image_object = Image(x_range_trunc, window_func=rect_window, signal=signal_vals_trunc, psi_obj=model_rec_fourier_psi, F=F)
    model_image_integral = model_image_object._evaluate_image()

    psi_l2 = np.linalg.norm(build_psi_values(kpsi_values, model_output_complex, x_range_trunc) - 
                            build_psi_values(kpsi_values, psi_coeffs_vals, x_range_trunc))

    islr_avg_nn = compute_islr(np.abs(model_image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
    islr_avg_grad = compute_islr(np.abs(image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)

    return islr_avg_nn, islr_avg_grad, psi_l2

def main():
    with open(MODEL_WEIGHTS_PATH, 'rb') as f:
        params = pickle.load(f)

    model = ConfigurableModel(architecture=[1093, 328, 963, 188, 514], activation_fn=jax.numpy.tanh)

    nn_islr_vals = []
    grad_islr_vals = []
    psi_l2_vals = []

    for i in range(NUM_COMPARISON_SAMPLES):
        sample_idx = np.random.randint(0, 100)
        islr_nn, islr_grad, psi_l2 = calculate_islr(model, params, sample_idx)
        nn_islr_vals.append(islr_nn)
        grad_islr_vals.append(islr_grad)
        psi_l2_vals.append(psi_l2)
        print(f"Sample {i+1}: ISLR (NN): {islr_nn}, ISLR (True): {islr_grad}, Psi L2: {psi_l2}")

    # Compute difference between ISLR NN and ISLR Grad
    islr_diff = np.array(nn_islr_vals) - np.array(grad_islr_vals)

    # Concatenate with Psi L2 on a new axis
    data_matrix = np.vstack((islr_diff, psi_l2_vals)).T  # Shape (N, 2)

    # Drop NaNs
    data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]

    plt.figure(figsize=(8, 6))
    plt.plot(data_matrix[:, 0], data_matrix[:, 1], 'o', label='ISLR Diff vs Psi l2')
    plt.xlabel("ISLR (NN - True)")
    plt.ylabel("Psi l2 Error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.savefig("islr_diff_vs_l2.png")
    plt.show()

if __name__ == "__main__":
    main()
