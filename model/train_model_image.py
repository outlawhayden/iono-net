import os
import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn
import numpy as np
import pandas as pd
import yaml
import random
from datetime import datetime
from flax.training import train_state
import optax
from model import ConfigurableModel
from tqdm import tqdm
import pickle
import json
import csv
from Helper import *
from Image import *
from Psi import *
from Optimize import *

jax.config.update("jax_enable_x64", True)

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights_90.pkl"

def trapezoid_integrate(y_vals, dx):
    return 0.5 * dx * jnp.sum(y_vals[:-1] + y_vals[1:])

# List available GPU devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")
if num_gpus > 1:
    print("Using multiple GPUs for training.")
    jax.config.update("jax_platform_name", "gpu")

with open("config_image.yaml", "r") as f:
    config = yaml.safe_load(f)

root_key = jax.random.PRNGKey(seed=config['seed'])

seed = config['seed']
np.random.seed(seed)            # <--- Seed numpy
random.seed(seed)               # <--- Seed python random
root_key = jax.random.PRNGKey(seed)  # <--- Already good
main_key, params_key, rng_key = jax.random.split(root_key, num=3)

architecture = config['model']['architecture']
activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

def linear_interp(x, xp, fp):
    def interp_batch(xi, xp, fp):
        idx = jnp.clip(jnp.searchsorted(xp, xi, side="right") - 1, 0, xp.shape[0] - 2)
        x0 = jnp.take(xp, idx)
        x1 = jnp.take(xp, idx + 1)
        y0 = jnp.take(fp, idx)
        y1 = jnp.take(fp, idx + 1)
        slope = (y1 - y0) / (x1 - x0 + 1e-8)
        return y0 + slope * (xi - x0)
    return jax.vmap(lambda xi: interp_batch(xi, xp, fp))(x)

def calculate_image_norm(x_range, signal_vals, preds_real, preds_imag, true_real, true_imag,
                         kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    signal_complex = signal_vals[:1441] + 1j * signal_vals[1441:]

    F_trunc = F/2
    mask_image = (x_range >= (x_range[0] + F_trunc)) & (x_range <= (x_range[-1] - F_trunc))
    x_trimmed = x_range[mask_image]
    signal_trimmed = signal_complex[mask_image]

    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

    def evaluate_single(y, real_coeffs, imag_coeffs):
        base = y + offsets
        real_interp = linear_interp(base, x_trimmed, jnp.real(signal_trimmed))
        imag_interp = linear_interp(base, x_trimmed, jnp.imag(signal_trimmed))
        signal_interp = real_interp + 1j * imag_interp
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_cos = jnp.sum(real_coeffs[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_sin = jnp.sum(imag_coeffs[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_vals = jnp.exp(1j * (psi_cos + psi_sin))
        integrand = waveform * signal_interp * psi_vals
        integral_real = trapezoid_integrate(jnp.real(integrand), DX)
        integral_imag = trapezoid_integrate(jnp.imag(integrand), DX)
        return (integral_real + 1j * integral_imag) / F

    image_vals_pred = vmap(lambda y: evaluate_single(y, preds_real, preds_imag))(x_trimmed)
    image_vals_true = vmap(lambda y: evaluate_single(y, true_real, true_imag))(x_trimmed)
    diff = image_vals_pred - image_vals_true
    return jnp.trapezoid(jnp.abs(diff) ** 2, x_trimmed)

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
label_matrix = label_df.map(convert_to_complex).to_numpy().T
data_matrix = data_df.map(convert_to_complex).to_numpy().T

label_matrix = normalize_complex_to_unit_range(label_matrix)
data_matrix = normalize_complex_to_unit_range(data_matrix)

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

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1])

test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading test dataset...")
    test_label_df = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
    test_data_df = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)
    test_label_matrix = test_label_df.map(convert_to_complex).to_numpy().T
    test_data_matrix = test_data_df.map(convert_to_complex).to_numpy().T
    test_label_matrix = normalize_complex_to_unit_range(test_label_matrix)
    test_data_matrix = normalize_complex_to_unit_range(test_data_matrix)

    
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

def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_image_diff, fourier_weight, image_diff_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    real_diffs = preds_real - true_real
    imag_diffs = preds_imag - true_imag
    sq_diffs = real_diffs ** 2 + imag_diffs ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))

    if add_image_diff:
        batch_size = inputs.shape[0]
        sample_indices = jax.random.choice(rng_key, batch_size, shape=(batch_size,), replace=False)
        def compute_single_image_diff(index):
            signal_sample = inputs[index]
            preds_real_sample = preds_real[index]
            preds_imag_sample = preds_imag[index]
            true_real_sample = true_real[index]
            true_imag_sample = true_imag[index]
            return calculate_image_norm(x_range, signal_sample, preds_real_sample, preds_imag_sample,
                                        true_real_sample, true_imag_sample,
                                        kpsi_values, ionoNHarm, F, dx, xi, zero_pad)
        loss_image = jnp.mean(jax.vmap(compute_single_image_diff)(sample_indices))
    else:
        loss_image = 0.0

    total_loss = fourier_weight * direct_loss + image_diff_weight * loss_image
    return total_loss, (direct_loss, loss_image)

gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
fourier_weight = config['training'].get('fourier_weight', 0.5)
image_diff_weight = 1.0 - fourier_weight

fixed_learning_rate = config['learning_rate']['fixed']
opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=fixed_learning_rate, weight_decay=l2_reg_weight)
)

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
with open("training_losses_image.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss", "Fourier Loss", "Image Loss"])


for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    batch_loss = 0.0
    batch_image_loss = 0.0
    batch_direct_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        (loss, (direct_loss, image_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, model, batch_signal, batch_coefficients,
            deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_image_diff=True,
            fourier_weight=fourier_weight, image_diff_weight=image_diff_weight
        )
        state = state.apply_gradients(grads=grads)
        batch_loss += loss
        batch_direct_loss += direct_loss
        batch_image_loss += image_loss / batch_signal.shape[0]

    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // config['training']['batch_size']
        for test_signal, test_coefficients in data_loader(test_dataset, config['training']['batch_size'], shuffle=False):
            total_test_loss, _ =  loss_fn(
                state.params, model, test_signal, test_coefficients,
                deterministic=True, rng_key=rng_key,
                ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_image_diff=True,
                fourier_weight=fourier_weight, image_diff_weight=image_diff_weight
            )
            test_loss += total_test_loss
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_image.csv", "a", newline='') as f:
        writer = csv.writer(f)
        avg_direct_loss = batch_direct_loss / num_batches
        avg_image_loss = batch_image_loss / num_batches
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss, avg_direct_loss, avg_image_loss])


final_weights_name = f"model_weights_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
