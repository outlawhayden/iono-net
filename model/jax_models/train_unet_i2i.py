# === Imports ===
# Standard libraries
import os                         # OS-level operations
import random                    # For setting random seeds
from datetime import datetime    # For timestamping model output

# Numerical and ML ecosystem
import numpy as np               # Core numerical library
import pandas as pd              # For reading CSV input files
import jax                       # Core JAX library
import jax.numpy as jnp          # JAX's NumPy-compatible API
from jax import vmap             # For vectorizing functions
import flax.linen as nn          # Flax neural network module API
from flax.training import train_state  # For managing model/optimizer state
import optax                     # Optimization library for JAX

# Progress bar
from tqdm import tqdm            # For showing training progress

# File I/O and serialization
import pickle                    # For saving model weights
import json                      # For reading metadata
import csv                       # For writing losses to file
import yaml                      # For reading config file

# Project-specific modules
from UNet1D_i2i import *         # Custom 1D U-Net model
from Helper import *             # Custom utilities (not shown)
from Image import *              # Image-domain tools (e.g., Fourier transforms)
from Psi import *                # Psi transform-related functions
from Optimize import *           # Optimization support

# Enable 64-bit precision for all computations
jax.config.update("jax_enable_x64", True)

# Show full tracebacks for easier debugging
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# === GPU Setup ===
# Detect available devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")
# Force use of GPU backend if multiple are available
if num_gpus > 1:
    print("Using multiple GPUs for training.")
    jax.config.update("jax_platform_name", "gpu")

# === Load configuration file ===
# Read experiment configuration from YAML file
with open("/home/houtlaw/iono-net/model/config_unet_i2i.yaml", "r") as f:
    config = yaml.safe_load(f)

# === RNG Seeding for reproducibility ===
# Set seeds for all RNGs
seed = config['seed']
np.random.seed(seed)
random.seed(seed)
root_key = jax.random.PRNGKey(seed)
main_key, params_key, rng_key = jax.random.split(root_key, num=3)

# === Load file paths from config ===
# Pull all path variables from config
label_file_path = config['paths']['label_data_file_path']
data_file_path = config['paths']['data_file_path']
x_range_file_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_values_path = config['paths']['kpsi_file_path']

# === Load metadata ===
# Load imaging parameters and Psi kernel from files
with open(setup_path) as f:
    setup = json.load(f)
F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
kpsi_values = pd.read_csv(kpsi_values_path).values
dx = 0.25  # step size for sampling

# === Utility functions ===
# Safely convert stringified complex numbers from CSV
def convert_to_complex(s):
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace('i', 'j'))

# Split a complex-valued array into real and imag channels for NN input
def stack_real_imag_as_channels(complex_array):
    real_part = complex_array.real[..., np.newaxis]
    imag_part = complex_array.imag[..., np.newaxis]
    return np.concatenate([real_part, imag_part], axis=-1)

# === Load and normalize training data ===
# Read raw complex signals and convert them to normalized real-imag arrays
focused_image_matrix_raw = pd.read_csv(label_file_path).map(convert_to_complex).to_numpy().T
data_matrix_raw = pd.read_csv(data_file_path).map(convert_to_complex).to_numpy().T
x_range = pd.read_csv(x_range_file_path).iloc[:,0].values

# Normalize input data (zero mean, unit std) and convert to (real, imag) channels
data_mean = np.mean(data_matrix_raw)
data_std = np.std(data_matrix_raw)
data_matrix_norm = (data_matrix_raw - data_mean) / data_std
data_matrix = stack_real_imag_as_channels(data_matrix_norm.T)

# Normalize target data (same process)
focused_mean = np.mean(focused_image_matrix_raw)
focused_std = np.std(focused_image_matrix_raw)
focused_image_matrix_norm = (focused_image_matrix_raw - focused_mean) / focused_std
focused_image_matrix = stack_real_imag_as_channels(focused_image_matrix_norm.T)

# === Load and normalize test data ===
test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading Test Dataset")
    test_label_matrix_raw = pd.read_csv(config["paths"]["test_label_file_path"]).map(convert_to_complex).to_numpy().T
    test_label_matrix_norm = (test_label_matrix_raw - focused_mean) / focused_std
    test_label_matrix = stack_real_imag_as_channels(test_label_matrix_norm.T)

    test_data_matrix_raw = pd.read_csv(config["paths"]["test_data_file_path"]).map(convert_to_complex).to_numpy().T
    test_data_matrix_norm = (test_data_matrix_raw - data_mean) / data_std
    test_data_matrix = stack_real_imag_as_channels(test_data_matrix_norm.T)
    test_dataset = 'yes'
else:
    print("No Test Dataset Loaded")

# === Pad signals to multiple of 2^depth ===
model_config = config["model_config"]
batch_size = config["training"]["batch_size"]
depth = len(model_config["down_channels"])
divisor = 2 ** depth  # required signal length multiple for U-Net downsampling

signal_length = data_matrix.shape[1]
remainder = signal_length % divisor

# If necessary, pad the signal so it divides evenly into UNet blocks
if remainder != 0:
    pad_total = divisor - remainder
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    print(f"Padding signal from length {signal_length} to {signal_length + pad_total} (pad_left={pad_left}, pad_right={pad_right})")

    def pad_signal(x):
        return np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))

    data_matrix = pad_signal(data_matrix)
    focused_image_matrix = pad_signal(focused_image_matrix)

    if test_dataset is not None:
        if test_data_matrix.shape[1] % divisor != 0:
            pad_total = divisor - (test_data_matrix.shape[1] % divisor)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            print(f"Padding test signal from length {test_data_matrix.shape[1]} to {test_data_matrix.shape[1] + pad_total}")
            test_data_matrix = pad_signal(test_data_matrix)
            test_label_matrix = pad_signal(test_label_matrix)
        test_dataset = list(zip(test_data_matrix, test_label_matrix))

# Pack training data
dataset = list(zip(data_matrix, focused_image_matrix))

# === Instantiate U-Net model ===
model = UNet1D_i2i(
    down_channels=model_config["down_channels"],
    bottleneck_channels=model_config["bottleneck_channels"],
    up_channels=model_config["up_channels"],
    output_dim=2  # output is (real, imag)
)

# Initialize model weights with dummy input
x_dummy = jnp.ones((batch_size, data_matrix.shape[1], 2))
variables = model.init(params_key, x_dummy)
print("Model initialized.")
print("Param structure:", jax.tree_util.tree_map(lambda x: x.shape, variables))

# === Data loader generator ===
# Yields batches of data from the dataset
def data_loader(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        batch_image = [dataset[i][0] for i in batch_indices]
        batch_labels = [dataset[i][1] for i in batch_indices]
        yield jnp.array(batch_image), jnp.array(batch_labels)

# === Loss function definition ===
def loss_fn(params, model, inputs, true_images, deterministic, rng_key, amp_weight, l4_weight):
    preds = model.apply({'params': params}, inputs)

    # Standard complex L2 loss
    real_diffs = preds[..., 0] - true_images[..., 0]
    imag_diffs = preds[..., 1] - true_images[..., 1]
    sq_diffs = real_diffs**2 + imag_diffs**2

    # Weight error by amplitude to prioritize high-magnitude areas
    amp_true = jnp.sqrt(true_images[..., 0]**2 + true_images[..., 1]**2)
    weight = amp_true / (jnp.max(amp_true, axis=1, keepdims=True) + 1e-8)
    weight = jnp.clip(weight, 1e-3, 1.0)

    weighted_sq_diffs = weight * sq_diffs
    weighted_loss = jnp.mean(jnp.sum(weighted_sq_diffs, axis=1))

    # Amplitude regularization  term
    amp_pred = jnp.sqrt(preds[..., 0]**2 + preds[..., 1]**2)
    amp_loss = jnp.mean(jnp.sum((amp_pred - amp_true)**2, axis=1))

    l4_loss = jnp.mean(jnp.sum(jnp.abs(real_diffs + 1j * imag_diffs) ** 4, axis=1))
    total_loss = weighted_loss + amp_weight * amp_loss + l4_loss * l4_weight

    return total_loss, weighted_loss

# === Optimizer ===
gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 0.01)
l4_weight = config['training'].get('l4_weight', 0)
amp_weight = config['training'].get('amp_weight', 0)
fixed_learning_rate = config['learning_rate']['fixed']

# Use AdamW with gradient clipping
opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=fixed_learning_rate, weight_decay=l2_reg_weight)
)

# === Initialize training state ===
state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

# === Training loop ===
loss_history = []
test_loss_history = []

# Initialize loss logging CSV
with open("training_losses_unet_i2i.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

# Main epoch loop
for epoch in tqdm(range(config["optimizer"]["num_epochs"]), desc="Training", position=0):
    batch_loss = 0.0
    num_batches = int(np.ceil(len(dataset) / batch_size))

    for batch_image, batch_labels in data_loader(dataset, batch_size):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, model, batch_image, batch_labels, deterministic=False, rng_key=subkey, amp_weight = amp_weight, l4_weight = l4_weight)
        state = state.apply_gradients(grads=grads)
        batch_loss += loss[0]

    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    # Evaluate on test data
    if test_dataset is not None:
        test_loss = 0.0
        test_batches = max(1, len(test_dataset) // batch_size)
        for test_image, test_coefficients in data_loader(test_dataset, batch_size, shuffle=False):
            total_test_loss, _ = loss_fn(state.params, model, test_image, test_coefficients, deterministic=True, rng_key=rng_key, l4_weight = l4_weight, amp_weight = amp_weight)
            test_loss += total_test_loss
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
        print(test_loss_history)
    else:
        avg_test_loss = None

    # Write epoch losses to CSV
    with open("training_losses_unet_i2i.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])

# Save final trained weights
final_weights_name = f"unet_weights_i2i_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
