import os
import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from flax.training import train_state
import optax
from model2 import ConfigurableModel
from tqdm import tqdm
import pickle
import json
import csv
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights_90.pkl"

# === Load Config ===
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

root_key = jax.random.PRNGKey(seed=config['seed'])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(architecture=config['model']['architecture'], activation_fn=activation_fn)

# === Helpers ===
def convert_to_complex(s):
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace('i', 'j'))

def compute_complex_mean_std(matrix):
    real_mean = np.mean(matrix.real)
    real_std = np.std(matrix.real)
    imag_mean = np.mean(matrix.imag)
    imag_std = np.std(matrix.imag)
    return real_mean, real_std, imag_mean, imag_std

def standardize_complex_matrix(matrix, real_mean, real_std, imag_mean, imag_std):
    real = (matrix.real - real_mean) / (real_std + 1e-8)
    imag = (matrix.imag - imag_mean) / (imag_std + 1e-8)
    return real + 1j * imag

def split_complex_to_channel(complex_array):
    real = complex_array.real
    imag = complex_array.imag
    return np.stack([real, imag], axis=-1)  # shape (N, L, 2)

# === Load Raw Training Data ===
raw_label_matrix = pd.read_csv(config['paths']['label_data_file_path'], dtype=str).map(convert_to_complex).to_numpy().T
raw_data_matrix = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str).map(convert_to_complex).to_numpy().T

# === Compute Dataset-wide Stats (inputs only) ===
data_real_mean, data_real_std, data_imag_mean, data_imag_std = compute_complex_mean_std(raw_data_matrix)

# === Apply Standardization to Input Only ===
data_matrix = standardize_complex_matrix(raw_data_matrix, data_real_mean, data_real_std, data_imag_mean, data_imag_std)
label_matrix = raw_label_matrix  # No standardization applied

# === Format for Model Input ===
data_matrix_split = split_complex_to_channel(data_matrix)
label_matrix_split = split_complex_to_channel(label_matrix)
dataset = list(zip(data_matrix_split, label_matrix_split))

# === Load X Range and Metadata ===
x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
with open(config['paths']['setup_file_path']) as f:
    setup = json.load(f)
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values
dx = 0.25
zero_pad = 50

# === Optional Test Data ===
test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    test_raw_label = pd.read_csv(config['paths']['test_label_file_path'], dtype=str).map(convert_to_complex).to_numpy().T
    test_raw_data = pd.read_csv(config['paths']['test_data_file_path'], dtype=str).map(convert_to_complex).to_numpy().T

    test_data_matrix = standardize_complex_matrix(test_raw_data, data_real_mean, data_real_std, data_imag_mean, data_imag_std)
    test_label_matrix = test_raw_label  # No standardization

    test_label_matrix_split = split_complex_to_channel(test_label_matrix)
    test_data_matrix_split = split_complex_to_channel(test_data_matrix)
    test_dataset = list(zip(test_data_matrix_split, test_label_matrix_split))

# === Save Stats for Inference (input only) ===
standardization_stats = {
    "data_real_mean": float(data_real_mean),
    "data_real_std": float(data_real_std),
    "data_imag_mean": float(data_imag_mean),
    "data_imag_std": float(data_imag_std)
}
with open("standardization_stats.json", "w") as f:
    json.dump(standardization_stats, f)
# === Format for Model Input ===
data_matrix_split = split_complex_to_channel(data_matrix)
label_matrix_split = split_complex_to_channel(label_matrix)
dataset = list(zip(data_matrix_split, label_matrix_split))

# === Load X Range and Metadata ===
x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
with open(config['paths']['setup_file_path']) as f:
    setup = json.load(f)
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values
dx = 0.25
zero_pad = 50

# === Optional Test Data ===
test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    test_raw_label = pd.read_csv(config['paths']['test_label_file_path'], dtype=str).map(convert_to_complex).to_numpy().T
    test_raw_data = pd.read_csv(config['paths']['test_data_file_path'], dtype=str).map(convert_to_complex).to_numpy().T

    test_label_matrix = test_raw_label
    test_data_matrix = standardize_complex_matrix(test_raw_data, data_real_mean, data_real_std, data_imag_mean, data_imag_std)

    test_label_matrix_split = split_complex_to_channel(test_label_matrix)
    test_data_matrix_split = split_complex_to_channel(test_data_matrix)
    test_dataset = list(zip(test_data_matrix_split, test_label_matrix_split))

# === Save Stats for Inference ===
standardization_stats = {
    "data_real_mean": float(data_real_mean),
    "data_real_std": float(data_real_std),
    "data_imag_mean": float(data_imag_mean),
    "data_imag_std": float(data_imag_std)
}
with open("standardization_stats.json", "w") as f:
    json.dump(standardization_stats, f)

# === Data Loader ===
def data_loader(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, dataset_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_signal = [dataset[i][0] for i in batch_indices]
        batch_coefficients = [dataset[i][1] for i in batch_indices]
        yield jnp.array(batch_signal), jnp.array(batch_coefficients)

# === Loss ===
def calculate_l4_norm(x_range, signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    signal_complex = signal_vals[:, 0] + 1j * signal_vals[:, 1]
    signal_trimmed = signal_complex[4 * zero_pad : -4 * zero_pad]
    x_trimmed = x_range[4 * zero_pad : -4 * zero_pad]
    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

    def evaluate_single(y):
        base = y + offsets
        real_interp = jnp.interp(base, x_trimmed, jnp.real(signal_trimmed))
        imag_interp = jnp.interp(base, x_trimmed, jnp.imag(signal_trimmed))
        signal_interp = real_interp + 1j * imag_interp
        window = jnp.ones_like(base)
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_cos = jnp.sum(preds_real[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_sin = jnp.sum(preds_imag[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_vals = jnp.exp(1j * (psi_cos + psi_sin))
        integrand = waveform * signal_interp * window * psi_vals
        return jnp.trapezoid(jnp.real(integrand), dx=DX) + 1j * jnp.trapezoid(jnp.imag(integrand), dx=DX)

    image_vals = vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4)

def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds = preds.reshape((preds.shape[0], 6, 2))
    true_coeffs = true_coeffs.reshape((true_coeffs.shape[0], 6, 2))

    real_diffs = preds[..., 0] - true_coeffs[..., 0]
    imag_diffs = preds[..., 1] - true_coeffs[..., 1]
    sq_diffs = real_diffs ** 2 + imag_diffs ** 2

    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=-1))
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 2) * sq_diffs, axis=-1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 4) * sq_diffs, axis=-1))
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    if add_l4:
        batch_size = inputs.shape[0]
        sample_indices = jax.random.choice(rng_key, batch_size, shape=(4,), replace=False)
        def compute_single_l4(index):
            signal_sample = inputs[index]
            preds_real = preds[index, :, 0]
            preds_imag = preds[index, :, 1]
            return calculate_l4_norm(x_range, signal_sample, preds_real, preds_imag, kpsi_values, ionoNHarm, F, dx, xi, zero_pad)
        loss_l4 = jnp.mean(jax.vmap(compute_single_l4)(sample_indices))
    else:
        loss_l4 = 0.0

    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4)

# === Training Setup ===
gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
l4_weight = config['training'].get('l4_reg_weight', 1e-3)
fourier_weight = config['training'].get('fourier_weight', 1e-3)
fourier_d1_weight = config['training'].get('fourier_d1_weight', 1e-3)
fourier_d2_weight = config['training'].get('fourier_d2_weight', 1e-3)

opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=config['learning_rate']['fixed'], weight_decay=l2_reg_weight)
)

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1], 2)
variables = model.init(main_key, jnp.ones(input_shape), deterministic=True)

if os.path.exists(cached_weights_path):
    print("Skipping loading weights due to format change.")
    params = variables['params']
else:
    params = variables['params']

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

# === Training Loop ===
loss_history = []
test_loss_history = []
with open("training_losses_full_2.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training"):
    batch_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']

    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coefficients,
            deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
            l2_reg_weight=l2_reg_weight, fourier_weight=fourier_weight,
            fourier_d1_weight=fourier_d1_weight, fourier_d2_weight=fourier_d2_weight,
            l4_weight=l4_weight
        )
        state = state.apply_gradients(grads=grads)
        batch_loss += loss

    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // config['training']['batch_size']
        for test_signal, test_coefficients in data_loader(test_dataset, config['training']['batch_size'], shuffle=False):
            test_loss += loss_fn(
                state.params, model, test_signal, test_coefficients,
                deterministic=True, rng_key=rng_key,
                ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=False,
                l2_reg_weight=l2_reg_weight, fourier_weight=fourier_weight,
                fourier_d1_weight=fourier_d1_weight, fourier_d2_weight=fourier_d2_weight,
                l4_weight=l4_weight
            )
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_full_2.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])

    print(f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}" +
          (f", Test Loss = {avg_test_loss:.6f}" if avg_test_loss else ""))

# === Save final weights ===
final_weights_name = f"model_weights_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
