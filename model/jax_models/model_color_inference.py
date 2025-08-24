import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from model_color import ConfigurableModel
import json
#from Helper import convert_to_complex, normalize_complex_to_unit_range, split_complex_to_imaginary, stack_real_imag_as_channels

# === Load Config ===
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace('i', 'j'))
    
def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    normalized = matrix / amp_max
    return normalized.real + 1j * normalized.imag

def normalize_zscore(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    print("mean = ", mean)
    return (matrix - mean) / (std + 1e-8)

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def stack_real_imag_as_channels(complex_array):
    real = complex_array.real[..., np.newaxis]
    imag = complex_array.imag[..., np.newaxis]
    return np.concatenate([real, imag], axis=-1)  # shape: (n_samples, signal_length, 2)

# === Load Ψ parameters ===
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()
setup = json.load(open(config['paths']['setup_file_path']))
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]

# === Load model ===
architecture = config['model']['architecture']
activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

with open("/home/houtlaw/iono-net/model/model_weights_color_20250607_023716.pkl", "rb") as f:
    params = pickle.load(f)

# === Load test data ===
label_df = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
data_df = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)

label_matrix = normalize_complex_to_unit_range(label_df.map(convert_to_complex).to_numpy().T)
data_matrix = normalize_complex_to_unit_range(data_df.map(convert_to_complex).to_numpy().T)

label_split = split_complex_to_imaginary(label_matrix)
data_split = stack_real_imag_as_channels(data_matrix)

# === Pick a random sample ===
idx = np.random.randint(len(data_split))
input_sample = jnp.array(data_split[idx:idx+1])  # Shape: (1, length, 2)
true_coeffs = label_split[idx]
true_real, true_imag = true_coeffs[:6], true_coeffs[6:]

# === Run model ===
pred_output = model.apply({'params': params}, input_sample, deterministic=True)
pred_real, pred_imag = np.array(pred_output[0, :6]), np.array(pred_output[0, 6:])

# === Define Ψ(x) function ===
def compute_psi_wave(x, real_coeffs, imag_coeffs):
    x = x[None, :]  # shape (1, N)
    phase = (real_coeffs[:, None] * np.cos(np.outer(kpsi_values, x))).sum(0) \
          + (imag_coeffs[:, None] * np.sin(np.outer(kpsi_values, x))).sum(0)
    return np.exp(1j * phase)

# === Evaluate Ψ on uniform grid ===
x_grid = np.linspace(-5, 5, 1000)
psi_true = compute_psi_wave(x_grid, true_real, true_imag)
psi_pred = compute_psi_wave(x_grid, pred_real, pred_imag)

# === Plot ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_grid, np.real(psi_true), label='Re Ψ True', linestyle='-')
plt.plot(x_grid, np.imag(psi_true), label='Im Ψ True', linestyle='--')
plt.title("True Ψ(x)")
plt.xlabel("x")
plt.ylabel("Ψ(x)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_grid, np.real(psi_pred), label='Re Ψ Pred', linestyle='-')
plt.plot(x_grid, np.imag(psi_pred), label='Im Ψ Pred', linestyle='--')
plt.title("Predicted Ψ(x)")
plt.xlabel("x")
plt.ylabel("Ψ(x)")
plt.legend()

plt.suptitle(f"Ψ comparison at sample index {idx}")
plt.tight_layout()

plt.savefig("color_figure.png")