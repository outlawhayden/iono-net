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
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "../../model"))
if model_dir not in sys.path:
    sys.path.append(model_dir)

print(f"Added {model_dir} to sys.path")
from Helper import *
from Image import *
from Psi import *
from Optimize import *
from conv_model import ConvModel  # Import the convolutional model instead of FCNN

# Matplotlib settings
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (30, 8)
plt.rcParams["savefig.dpi"] = 300

# Parameters
SAMPLE_IDX = 2
DX = 0.25
ISLR_RADIUS = 5
ISLR_RADIUS_RATIO = 0.6
ISLR_MAIN_LOBE_WIDTH = 0.75

# File paths
DATA_DIR = "/home/houtlaw/iono-net/data/baselines/10k_lownoise"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250206_104914.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250206_104914.json"
SCATTERER_PATH_RELNOISE = f"{DATA_DIR}/test_nuStruct_withSpeckle_20250206_104911.csv"
SIGNAL_PATH_RELNOISE = f"{DATA_DIR}/test_uscStruct_vals_20250206_104913.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250206_104914.csv"
PSI_COEFFS_PATH_RELNOISE = f"{DATA_DIR}/test_compl_ampls_20250206_104913.csv"
MODEL_WEIGHTS_PATH = "/home/houtlaw/iono-net/model/most_recent_conv_weights.pkl"

# Helper Functions
def convert_to_complex(s):
    s = str(s)
    return 0 if s == "NaNNaNi" else complex(s.replace('i', 'j'))

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
    for peak in peaks:
        inner_indices = [i for i, x in enumerate(x_vals) if np.abs(x - peak) <= main_lobe_width]
        outer_indices = [i for i, x in enumerate(x_vals) if main_lobe_width < np.abs(x - peak) <= (radius * radius_ratio)]
        inner_integral = integrate.trapezoid(image_integral[inner_indices], x=x_vals[inner_indices], dx=dx)
        outer_integral = integrate.trapezoid(image_integral[outer_indices], x=x_vals[outer_indices], dx=dx)
        islr = 10 * np.log10(outer_integral / inner_integral) if inner_integral != 0 else 0
        islrs.append(islr)
    return np.mean(islrs)

def main():
    x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values
    with open(SETUP_PATH) as f:
        setup = json.load(f)

    true_scatterers = pd.read_csv(SCATTERER_PATH_RELNOISE).map(convert_to_complex).iloc[:, SAMPLE_IDX].map(np.abs).values
    signal_data = pd.read_csv(SIGNAL_PATH_RELNOISE).map(convert_to_complex).T.iloc[SAMPLE_IDX].values
    signal_vals = np.vstack((x_range, signal_data))
    kpsi_values = pd.read_csv(KPSI_PATH).values
    psi_coeffs_df = pd.read_csv(PSI_COEFFS_PATH_RELNOISE).T
    psi_coeffs_df = psi_coeffs_df.map(lambda x: complex(x.replace('i', 'j')))
    psi_coeffs_vals = psi_coeffs_df.iloc[SAMPLE_IDX].values

    signal_array = signal_vals[1, :]
    nonzero_indices = np.where(np.abs(signal_array) > 1e-12)[0]
    start_idx, end_idx = nonzero_indices[0], nonzero_indices[-1]
    x_range_trunc = x_range[start_idx : end_idx + 1]
    signal_vals_trunc = np.vstack((x_range_trunc, signal_array[start_idx : end_idx + 1]))

    F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
    cos_coeffs = [j.real for j in psi_coeffs_vals]
    sin_coeffs = [-j.imag for j in psi_coeffs_vals]
    rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
    rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)
    image_object = Image(x_range_trunc, rect_window, signal_vals_trunc, rec_fourier_psi, F=F, dx=DX)
    image_integral = image_object._evaluate_image()
    plt.plot(x_range, true_scatterers, 'orange', lw=3)
    plt.plot(x_range_trunc, np.abs(image_integral) / DX, lw=2)
    plt.title("Image Integral")
    add_plot_subtitle(setup)
    plt.legend(["True Point Scatterers", "Image Integral"])
    plt.savefig("image_integral.png")

    with open(MODEL_WEIGHTS_PATH, 'rb') as f:
        params = pickle.load(f)

    model_input = split_complex_to_imaginary(signal_array[start_idx:end_idx + 1])
    model_input = model_input.reshape(1, -1)

    model = ConvModel(conv_channels=[16, 32], dense_layers=[64], activation_fn=jnp.tanh, dropout_rate=0.0)
    variables = model.init(jax.random.PRNGKey(0), jnp.array(model_input), deterministic=True)
    preds = model.apply(variables, jnp.array(model_input), deterministic=True)[0]
    preds_real, preds_imag = preds[:len(preds)//2], preds[len(preds)//2:]
    model_output_complex = preds_real + 1j * preds_imag

    model_cos_coeffs = [j.real for j in model_output_complex]
    model_sin_coeffs = [-j.imag for j in model_output_complex]
    model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
    model_rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)
    model_image_object = Image(x_range_trunc, rect_window, signal_vals_trunc, model_rec_fourier_psi, F=F, dx=DX)
    model_image_integral = model_image_object._evaluate_image()

    plt.figure()
    plt.plot(x_range, true_scatterers, 'orange', lw=3)
    plt.plot(x_range_trunc, np.abs(model_image_integral) / DX, lw=2)
    plt.title("Image Integral (NN) Inference")
    add_plot_subtitle(setup)
    plt.legend(["True Point Scatterers", "Image Integral (NN)"])
    plt.savefig("neural_image_integral.png")

    model_psi_vals = build_psi_values(kpsi_values, model_output_complex, x_range_trunc)
    true_psi_vals = build_psi_values(kpsi_values, psi_coeffs_vals, x_range_trunc)
    print("True Psi Values:", true_psi_vals)
    print("Model Psi Values:", model_psi_vals)

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

    islr_avg_nn = compute_islr(np.abs(model_image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
    print("ISLR Average (NN):", islr_avg_nn)
    islr_avg_grad = compute_islr(np.abs(image_integral) / DX, true_scatterers, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
    print("ISLR Average (Classic):", islr_avg_grad)

if __name__ == "__main__":
    main()
