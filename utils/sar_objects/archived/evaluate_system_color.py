# Imports and Setup
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flax.linen as nn

from matplotlib import rcParams
import jax
import yaml
import jax.numpy as jnp

# Custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "../../model_color"))
if model_dir not in sys.path:
    sys.path.append(model_dir)

from Helper import *
from Image import *
from Psi import *
from model_color import ConfigurableModel

plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (30, 8)
plt.rcParams["savefig.dpi"] = 300

# Parameters
SAMPLE_IDX = 2
DX = 0.25
ISLR_RADIUS = 5
ISLR_RADIUS_RATIO = 0.6
ISLR_MAIN_LOBE_WIDTH = 0.75

# Paths
DATA_DIR = "/home/houtlaw/iono-net/data/baselines/smaller_dataset"
MODEL_WEIGHTS_PATH = "/home/houtlaw/iono-net/model/model_weights_color_smaller_20250611_192528.pkl"
CONFIG_PATH = "/home/houtlaw/iono-net/model/config_simple.yaml"

# Load Config

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


label_file_path = config['paths']['label_data_file_path']
data_file_path = config['paths']['signal_data_file_path']
x_range_file_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_path = config['paths']['kpsi_file_path']

# Utility Functions
def convert_to_complex(s):
    s = str(s)
    return 0 if s == "NaNNaNi" else complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    normalized = matrix / amp_max
    return normalized.real + 1j * normalized.imag

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def stack_real_imag_as_channels(complex_array):
    real = complex_array.real[..., np.newaxis]
    imag = complex_array.imag[..., np.newaxis]
    return np.concatenate([real, imag], axis=-1)

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
        inner = (np.abs(x_vals - peak) <= main_lobe_width)
        outer = (np.abs(x_vals - peak) > main_lobe_width) & (np.abs(x_vals - peak) <= (radius * radius_ratio))
        inner_int = np.trapz(image_integral[inner], x=x_vals[inner], dx=dx)
        outer_int = np.trapz(image_integral[outer], x=x_vals[outer], dx=dx)
        islr = 10 * np.log10(outer_int / inner_int) if inner_int != 0 else 0
        islrs.append(islr)
    return np.mean(islrs)

# Load Data
x_range = pd.read_csv(x_range_file_path).iloc[:, 0].values
kpsi_values = pd.read_csv(kpsi_path).values.squeeze()
signal_df = pd.read_csv(data_file_path, dtype=str)
label_df = pd.read_csv(label_file_path, dtype=str)
setup = json.load(open(setup_path))

signal_matrix = normalize_complex_to_unit_range(signal_df.map(convert_to_complex).to_numpy())
label_matrix = label_df.map(convert_to_complex).to_numpy().T

# Prepare Input and Ground Truth
signal_tensor = stack_real_imag_as_channels(signal_matrix)
sample_input = jnp.expand_dims(signal_tensor[SAMPLE_IDX], axis=0)
true_coeffs = label_matrix[SAMPLE_IDX]

# Load Model
architecture = config['model']['architecture']
activation_name = config['model']['activation']
if hasattr(jnp, activation_name):
    activation_fn = getattr(jnp, activation_name)
elif hasattr(nn, activation_name):
    activation_fn = getattr(nn, activation_name)
else:
    raise ValueError(f"Activation function '{activation_name}' not found.")
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)
with open(MODEL_WEIGHTS_PATH, "rb") as f:
    params = pickle.load(f)

# Run Inference
model_output = model.apply({'params': params}, sample_input, deterministic=True)
model_output = jnp.squeeze(model_output)
n_coeffs = model_output.shape[0] // 2
pred_coeffs = model_output[:n_coeffs] + 1j * model_output[n_coeffs:]


print("OUT")
print(true_coeffs)
print(pred_coeffs)


# Evaluate Ψ
x_range_trunc = x_range  # Adjust this if trimming is required
true_psi_vals = build_psi_values(kpsi_values, true_coeffs, x_range_trunc)
pred_psi_vals = build_psi_values(kpsi_values, pred_coeffs, x_range_trunc)

plt.figure(figsize=(12, 6))
plt.plot(x_range_trunc, true_psi_vals, label="True Ψ", lw=4)
plt.plot(x_range_trunc, pred_psi_vals, label="Predicted Ψ")
plt.title("Ψ Comparison")
plt.xlabel("x")
plt.ylabel("Ψ(x)")
plt.grid(True)
plt.legend()
add_plot_subtitle(setup)
plt.savefig("psi_comparison_eval.png")
print("Saved Ψ comparison plot.")

# Compute Image from Predicted Coeffs
cos_coeffs = pred_coeffs.real
sin_coeffs = -pred_coeffs.imag
F, Nharm, xi = setup['F'], setup['ionoNharm'], setup['xi']
signal_for_eval = np.stack([x_range, signal_matrix[SAMPLE_IDX]])

rec_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, Nharm)
rec_psi.cache_psi(x_range_trunc, F, DX, xi)
image_obj = Image(x_range_trunc, rect_window, signal_for_eval, rec_psi, F)
image_integral = image_obj._evaluate_image()

# Plot image
true_scatterers = np.abs(label_matrix[SAMPLE_IDX])
plt.figure()
plt.plot(x_range, true_scatterers, 'orange', lw=3)
plt.plot(x_range_trunc, np.abs(image_integral) / DX, lw=2)
plt.title("Image Integral (NN)")
plt.legend(["True Scatterers", "Image (NN)"])
add_plot_subtitle(setup)
plt.savefig("neural_image_integral_eval.png")

# Compute ISLR
islr_nn = compute_islr(np.abs(image_integral) / DX, true_scatterers, x_range_trunc,
                       ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
print("ISLR (NN):", islr_nn)
