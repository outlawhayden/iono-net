import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax.training import train_state
import yaml
from UNet1D_i2i import *
from Helper import *
from Image import *
from Psi import *

jax.config.update("jax_enable_x64", True)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# === Load Config ===
with open("/home/houtlaw/iono-net/model/config_unet_i2i.yaml", "r") as f:
    config = yaml.safe_load(f)

# === File Paths ===
paths = config['paths']
SAMPLE_IDX = 2
DX = 0.25

# === Load Setup ===
with open(paths["setup_file_path"]) as f:
    setup = json.load(f)

F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]

# === Load Data ===
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def stack_real_imag_as_channels(complex_array):
    real_part = complex_array.real[..., np.newaxis]
    imag_part = complex_array.imag[..., np.newaxis]
    return np.concatenate([real_part, imag_part], axis=-1)

x_range = pd.read_csv(paths["x_range_file_path"]).iloc[:, 0].values
signal_df_full = pd.read_csv(paths["data_file_path"], dtype=str).map(convert_to_complex)
label_df_full = pd.read_csv(paths["label_data_file_path"], dtype=str).map(convert_to_complex)

signal_matrix = signal_df_full.to_numpy().T
label_matrix = label_df_full.to_numpy().T

# === Normalize and reshape ===
data_mean = np.mean(signal_matrix)
data_std = np.std(signal_matrix)
focused_mean = np.mean(label_matrix)
focused_std = np.std(label_matrix)

signal_matrix_norm = (signal_matrix - data_mean) / data_std
label_matrix_norm = (label_matrix - focused_mean) / focused_std

signal_tensor = stack_real_imag_as_channels(signal_matrix_norm.T)  # (n_samples, length, 2)
label_tensor = stack_real_imag_as_channels(label_matrix_norm.T)

sample_input = signal_tensor[SAMPLE_IDX:SAMPLE_IDX+1, :, :]  # shape (1, L, 2)
sample_true_label = label_tensor[SAMPLE_IDX:SAMPLE_IDX+1, :, :]

# === Pad Input to Match Model ===
depth = len(config["model_config"]["down_channels"])
divisor = 2 ** depth
signal_length = sample_input.shape[1]
pad = (divisor - (signal_length % divisor)) % divisor
pad_left, pad_right = pad // 2, pad - pad // 2

sample_input_padded = np.pad(sample_input, ((0, 0), (pad_left, pad_right), (0, 0)))

# === Load Model and Weights ===
model = UNet1D_i2i(
    down_channels=config["model_config"]["down_channels"],
    bottleneck_channels=config["model_config"]["bottleneck_channels"],
    up_channels=config["model_config"]["up_channels"],
    output_dim=2
)

params_path = "/home/houtlaw/iono-net/model/unet_weights_i2i_20250516_122404.pkl"
with open(params_path, "rb") as f:
    params = pickle.load(f)

# === Run Inference ===
predicted_label_norm = model.apply({'params': params}, jnp.array(sample_input_padded))

# === Remove padding and rescale ===
predicted_label_norm = np.array(predicted_label_norm)[0]  # (L_pad, 2)
if pad > 0:
    if pad_right > 0:
        predicted_label_norm = predicted_label_norm[pad_left:-pad_right, :]
    else:
        predicted_label_norm = predicted_label_norm[pad_left:, :]

# === Reconstruct Complex Signal ===
predicted_label_complex = predicted_label_norm[:, 0] + 1j * predicted_label_norm[:, 1]
predicted_label_complex *= focused_std
predicted_label_complex += focused_mean

true_label_complex = sample_true_label[0, :, 0] + 1j * sample_true_label[0, :, 1]
true_label_complex *= focused_std
true_label_complex += focused_mean

# === Plot Comparison ===
plt.figure(figsize=(12, 6))
plt.plot(x_range, np.abs(true_label_complex)[:-1], label="True |Image|", lw=3)
plt.plot(x_range, np.abs(predicted_label_complex)[:-1], label="Predicted |Image|", lw=2)
plt.xlabel("x")
plt.ylabel("Magnitude")
plt.title("Image Magnitude Comparison")
plt.legend()
plt.grid(True)
plt.savefig("unet_i2i_image_comparison.png")
plt.close()

# === Compute ISLR ===
SCATTERER_PATH = "/home/houtlaw/iono-net/data/baselines/10k_lownoise/test_nuStruct_withSpeckle_20250206_104911.csv"
true_scatterers = pd.read_csv(SCATTERER_PATH).map(convert_to_complex).iloc[:, SAMPLE_IDX].map(np.abs).values

def compute_islr(image_integral, known_scatterers, x_vals, radius, radius_ratio, main_lobe_width, dx):
    islrs = []
    peaks = [x_vals[i] for i, s in enumerate(known_scatterers) if s > 2]
    image_integral = image_integral ** 2
    for peak in peaks:
        inner_mask = np.abs(x_vals - peak) <= main_lobe_width
        outer_mask = (np.abs(x_vals - peak) > main_lobe_width) & (np.abs(x_vals - peak) <= (radius * radius_ratio))
        inner_integral = jnp.trapz(image_integral[inner_mask], x_vals[inner_mask], dx=dx)
        outer_integral = jnp.trapz(image_integral[outer_mask], x_vals[outer_mask], dx=dx)
        islr = 10 * jnp.log10(outer_integral / inner_integral) if inner_integral != 0 else 0
        islrs.append(islr)
    return float(np.mean(islrs))

#islr_nn = compute_islr(np.abs(predicted_label_complex), true_scatterers, x_range, radius=5, radius_ratio=0.6, main_lobe_width=0.75, dx=DX)
#print(f"ISLR Average (NN): {islr_nn:.2f} dB")
