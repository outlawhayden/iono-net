import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import yaml
import jax.numpy as jnp
from model_color import ConfigurableModel
from Helper import *
from Image import *
from Psi import *
from Optimize import *

# === CONFIGURATION ===
SAMPLE_IDX = 0
DX = 0.25

# Load configuration and paths
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)


label_path = config['paths']['test_label_file_path']
data_path = config['paths']['test_data_file_path']
x_range_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_path = config['paths']['kpsi_file_path']
weights_path = "/home/houtlaw/iono-net/model/model_weights_color_20250613_200047.pkl"

# === UTILS ===
def convert_to_complex(s):
    s = str(s)
    return 0 if s == "NaNNaNi" else complex(s.replace("i", "j"))

def normalize(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    return matrix / amp_max

def build_psi_values(kpsi, compl_ampls, x):
    return np.real(np.sum([
        compl_ampls[i] * np.exp(1j * kpsi[i] * x) for i in range(len(compl_ampls))
    ], axis=0))

# === LOAD DATA ===
x_range = pd.read_csv(x_range_path).iloc[:, 0].values
kpsi = pd.read_csv(kpsi_path).values[:, 0]
with open(setup_path) as f:
    setup = json.load(f)

F = setup["F"]
ionoNHarm = setup["ionoNharm"]
xi = setup["xi"]

label = pd.read_csv(label_path, dtype=str).map(convert_to_complex).to_numpy().T
signal = pd.read_csv(data_path, dtype=str).map(convert_to_complex).to_numpy().T

signal = normalize(signal)
label = normalize(label)

x = x_range
signal_sample = signal[SAMPLE_IDX]
label_sample = label[SAMPLE_IDX]

# === PREPARE INPUT ===
real = signal_sample.real[np.newaxis, ...]
imag = signal_sample.imag[np.newaxis, ...]
input_sample = np.concatenate([real, imag], axis=-1)  # shape: (1, len, 2)
flat_input = input_sample.reshape(1, -1)

# === LOAD MODEL ===
output_dim = 2 * len(kpsi)
architecture = config['model']['architecture']
activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

params = pickle.load(open(weights_path, 'rb'))
pred = model.apply({'params': params}, jnp.array(flat_input), deterministic=True).squeeze()
model_output_complex = pred[:len(kpsi)] + 1j * pred[len(kpsi):]

print("OUTPUT")
print(model_output_complex)


# Split model output into cosine and sine coefficients
model_cos_coeffs = [j.real for j in model_output_complex]
model_sin_coeffs = [-j.imag for j in model_output_complex]

# Create model Psi object
model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
model_rec_fourier_psi.cache_psi(x_range_trunc, F, DX, xi)

# Create model image object
model_image_object = Image(x_range_trunc, window_func=rect_window, signal=signal_vals_trunc, psi_obj=model_rec_fourier_psi, F=F)
model_image_integral = model_image_object._evaluate_image()

# Plot model results
plt.figure()
plt.plot(x_range, true_scatterers, 'orange', lw=3)
plt.plot(x_range_trunc, np.abs(model_image_integral) / DX, lw=2)
plt.title("Image Integral (NN) Inference")
add_plot_subtitle(setup)
plt.legend(["True Point Scatterers", "Image Integral (NN)"])
plt.savefig("neural_image_integral.png")

# Compute Psi Values and Plot
model_psi_vals = build_psi_values(kpsi_values, model_output_complex, x_range_trunc)
true_psi_vals = build_psi_values(kpsi_values, psi_coeffs_vals, x_range_trunc)

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

