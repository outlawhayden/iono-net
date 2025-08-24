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
from model import ConfigurableModel
from tqdm import tqdm
import pickle
import json
import csv
from flax.linen.initializers import variance_scaling
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights_90.pkl"

# List available GPU devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")
if num_gpus > 1:
    print("Using multiple GPUs for training.")
    jax.config.update("jax_platform_name", "gpu")

with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

root_key = jax.random.PRNGKey(seed=config['seed'])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

small_kernel_init = variance_scaling(1e-4, "fan_avg", "uniform")
zero_bias_init = nn.initializers.zeros

architecture = config['model']['architecture']
activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(
    architecture=architecture,
    activation_fn=activation_fn,
    kernel_init=small_kernel_init,
    bias_init=zero_bias_init
)
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

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

label_file_path = config['paths']['label_data_file_path']
data_file_path = config['paths']['signal_data_file_path']
x_range_file_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_values_path = config['paths']['kpsi_file_path']

label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)

label_matrix = normalize_complex_to_unit_range(label_df.map(convert_to_complex).to_numpy().T)
data_matrix = normalize_complex_to_unit_range(data_df.map(convert_to_complex).to_numpy().T)

x_range = pd.read_csv(x_range_file_path).iloc[:, 0].values

data_matrix_split = split_complex_to_imaginary(data_matrix)
label_matrix_split = split_complex_to_imaginary(label_matrix)
dataset = list(zip(data_matrix_split, label_matrix_split))

with open(setup_path) as f:
    setup = json.load(f)

F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
kpsi_values = pd.read_csv(kpsi_values_path).values

dx = 0.25
zero_pad = 50

test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading test dataset...")
    test_label_df = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
    test_data_df = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)
    test_label_matrix = normalize_complex_to_unit_range(test_label_df.map(convert_to_complex).to_numpy().T)
    test_data_matrix = normalize_complex_to_unit_range(test_data_df.map(convert_to_complex).to_numpy().T)
    test_label_matrix_split = split_complex_to_imaginary(test_label_matrix)
    test_data_matrix_split = split_complex_to_imaginary(test_data_matrix)
    test_dataset = list(zip(test_data_matrix_split, test_label_matrix_split))
else:
    print("No test dataset found.")
    

def data_loader(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        batch_signal = [dataset[i][0] for i in batch_indices]
        batch_coefficients = [dataset[i][1] for i in batch_indices]
        yield jnp.array(batch_signal), jnp.array(batch_coefficients)

def linear_interp(x, xp, fp):
    def interpolate_single(xi):
        idx = jnp.clip(jnp.searchsorted(xp, xi, side="right") - 1, 0, xp.shape[0] - 2)

        # Safely gather values without using arr[idx] directly
        x0 = jax.lax.dynamic_index_in_dim(xp, idx, keepdims=False)
        x1 = jax.lax.dynamic_index_in_dim(xp, idx + 1, keepdims=False)
        y0 = jax.lax.dynamic_index_in_dim(fp, idx, keepdims=False)
        y1 = jax.lax.dynamic_index_in_dim(fp, idx + 1, keepdims=False)

        slope = (y1 - y0) / (x1 - x0 + 1e-12)
        return y0 + slope * (xi - x0)

    return jax.vmap(interpolate_single)(x)

def calculate_l4_norm(x_range, signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    signal_complex = signal_vals[:1441] + 1j * signal_vals[1441:]
    signal_trimmed = signal_complex[4 * zero_pad : -4 * zero_pad]
    x_trimmed = x_range[4 * zero_pad : -4 * zero_pad]
    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

    def evaluate_single(y):
        base = y + offsets
        real_interp = linear_interp(base, x_trimmed, jnp.real(signal_trimmed))
        imag_interp = linear_interp(base, x_trimmed, jnp.imag(signal_trimmed))
        signal_interp = real_interp + 1j * imag_interp
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_cos = jnp.sum(preds_real[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_sin = jnp.sum(preds_imag[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_vals = jnp.exp(1j * (psi_cos - psi_sin))
        integrand = waveform * signal_interp * psi_vals
        integral = jnp.trapezoid(integrand, dx=DX)
        return integral / F

    image_vals = jax.vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4).real


def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    real_diffs = preds_real - true_real
    imag_diffs = preds_imag - true_imag
    sq_diffs = real_diffs ** 2 + imag_diffs ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 2) * sq_diffs, axis=1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 4) * sq_diffs, axis=1))
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))


    if add_l4:
        def compute_single_l4(signal_sample, real_pred, imag_pred):
            return calculate_l4_norm(x_range, signal_sample, real_pred, imag_pred, kpsi_values, ionoNHarm, F, dx, xi, zero_pad)
        loss_l4 = jnp.mean(jax.vmap(compute_single_l4)(inputs, preds_real, preds_imag))
    else:
        loss_l4 = 0.0

    print("L4 terms", loss_l4, loss_l4 * l4_weight)
    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4)

gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
l4_weight = config['training'].get('l4_reg_weight', 1e-3)
fourier_weight = config['training'].get('fourier_weight', 1e-3)
fourier_d1_weight = config['training'].get('fourier_d1_weight', 1e-3)
fourier_d2_weight = config['training'].get('fourier_d2_weight', 1e-3)

fixed_learning_rate = config['learning_rate']['fixed']  # e.g. 0.001 in your config
opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=fixed_learning_rate, weight_decay=l2_reg_weight)
)

# Use small initial weights and zero biases
small_kernel_init = variance_scaling(1e-4, "fan_avg", "uniform")
zero_bias_init = nn.initializers.zeros

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1])
variables = model.init(main_key, jnp.ones(input_shape), deterministic=True)

if os.path.exists(cached_weights_path):
    print("Loading saved model weights...")
    with open(cached_weights_path, "rb") as f:
        loaded_params = pickle.load(f)
    state = train_state.TrainState.create(apply_fn=model.apply, params=loaded_params, tx=opt)
else:
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

loss_history = []
test_loss_history = []
with open("training_losses_full.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    batch_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coefficients,
            deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
            l2_reg_weight=l2_reg_weight,
            fourier_weight=fourier_weight,
            fourier_d1_weight=fourier_d1_weight,
            fourier_d2_weight=fourier_d2_weight,
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
                ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
                l2_reg_weight=l2_reg_weight,
                fourier_weight=fourier_weight,
                fourier_d1_weight=fourier_d1_weight,
                fourier_d2_weight=fourier_d2_weight,
                l4_weight=l4_weight
            )
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_full.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])


    print(f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}, Test Loss = {avg_test_loss:.6f}" if avg_test_loss else f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}")

final_weights_name = f"model_weights_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
