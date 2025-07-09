import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import random
from UNet1D_i2i import *
from Helper import *
from Psi import *
from Image import *

# Load config
with open("/home/houtlaw/iono-net/model/config_unet_i2i.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed and JAX precision
seed = config['seed']
np.random.seed(seed)
jax.config.update("jax_enable_x64", True)

# Load data paths
label_file_path = config['paths']['test_label_file_path']
data_file_path = config['paths']['test_data_file_path']

# Convert function
def convert_to_complex(s): 
    if s == "NaNNaNi":
        return np.nan
    else:
        return complex(s.replace('i', 'j'))

def stack_real_imag_as_channels(complex_array):
    real_part = complex_array.real[..., np.newaxis]
    imag_part = complex_array.imag[..., np.newaxis]
    return np.concatenate([real_part, imag_part], axis=-1)

# Load and normalize data
focused_raw = pd.read_csv(label_file_path).map(convert_to_complex).to_numpy().T
unfocused_raw = pd.read_csv(data_file_path).map(convert_to_complex).to_numpy().T

# Normalize using training statistics
data_mean = np.mean(unfocused_raw)
data_std = np.std(unfocused_raw)
focused_mean = np.mean(focused_raw)
focused_std = np.std(focused_raw)

unfocused_norm = (unfocused_raw - data_mean) / data_std
focused_norm = (focused_raw - focused_mean) / focused_std

# Stack as channels
X = stack_real_imag_as_channels(unfocused_norm.T)
Y = stack_real_imag_as_channels(focused_norm.T)

# Pad to match training
depth = len(config['model_config']["down_channels"])
divisor = 2 ** depth
signal_len = X.shape[1]
remainder = signal_len % divisor
pad_left, pad_right = 0, 0
if remainder != 0:
    pad_total = divisor - remainder
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    X = np.pad(X, ((0,0), (pad_left, pad_right), (0,0)))
    Y = np.pad(Y, ((0,0), (pad_left, pad_right), (0,0)))

# Choose a random sample
idx = np.random.randint(0, X.shape[0])
x_sample = jnp.expand_dims(X[idx], axis=0)  # (1, length, 2)
y_true = Y[idx]  # (length, 2)

# Initialize and load model
model = UNet1D_i2i(
    down_channels=config["model_config"]["down_channels"],
    bottleneck_channels=config["model_config"]["bottleneck_channels"],
    up_channels=config["model_config"]["up_channels"],
    output_dim=2
)

# Load model weights
latest_weights_file = sorted([f for f in os.listdir(".") if f.startswith("unet_weights_i2i_") and f.endswith(".pkl")])[-1]
with open(latest_weights_file, "rb") as f:
    params = pickle.load(f)
print(f"Loaded model weights from {latest_weights_file}")

# Run inference
y_pred = model.apply({'params': params}, x_sample)[0]  # remove batch dim


# === Existing code remains unchanged ===

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
axs[0, 0].plot(y_true[:, 0], label="True Real")
axs[0, 0].plot(y_pred[:, 0], label="Predicted Real", linestyle="--")
axs[0, 0].set_title("Real Part")
axs[0, 0].legend()

axs[0, 1].plot(y_true[:, 1], label="True Imag")
axs[0, 1].plot(y_pred[:, 1], label="Predicted Imag", linestyle="--")
axs[0, 1].set_title("Imaginary Part")
axs[0, 1].legend()

axs[1, 0].plot(jnp.abs(y_true[:, 0] + 1j * y_true[:, 1]), label="True Amplitude")
axs[1, 0].plot(jnp.abs(y_pred[:, 0] + 1j * y_pred[:, 1]), label="Predicted Amplitude", linestyle="--")
axs[1, 0].set_title("Amplitude")
axs[1, 0].legend()

axs[1, 1].plot(jnp.angle(y_true[:, 0] + 1j * y_true[:, 1]), label="True Phase")
axs[1, 1].plot(jnp.angle(y_pred[:, 0] + 1j * y_pred[:, 1]), label="Predicted Phase", linestyle="--")
axs[1, 1].set_title("Phase")
axs[1, 1].legend()

plt.suptitle("Signal Components and Reconstruction, ionoOverAmpl = 0.5", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("i2i_figure.png")
plt.show()

# === New Image Plot ===
# Parameters
DX = config.get("dx", 0.25)
zero_pad = config.get("zero_pad", 50)

# Reconstruct images
true_complex = y_true[:, 0] + 1j * y_true[:, 1]
pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]

img_true = jnp.abs(true_complex/DX)
img_pred = jnp.abs(pred_complex/DX)

# Plot image comparison
plt.figure(figsize=(10, 4))
plt.plot(img_true, label="True Image")
plt.plot(img_pred, label="Predicted Image", linestyle="--")
plt.title("Reconstructed Image vs True, ionoOverAmpl = 0.5" )
plt.xlabel("Spatial Index")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("i2i_image_comparison.png")
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(img_true - img_pred, label="Image Error")
plt.title("Difference (Reconstructed Image - True), ionoOverAmpl = 0.5" )
plt.xlabel("Spatial Index")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("i2i_image_diffs.png")
plt.show()
