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
from conv_model import ComplexUNet1D
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
main_key, params_key, rng_key = jax.random.split(root_key, num=3)

model = ComplexUNet1D(
    depth=config['model'].get('depth', 3),
    base_features=config['model'].get('base_features', 32),
    kernel_size=config['model'].get('kernel_size', 3)
)

fixed_learning_rate = config['learning_rate']['fixed']
gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
fourier_weight = config['training'].get('fourier_weight', 1e-3)
image_diff_weight = config['training'].get('image_diff_weight', 1e-3)

opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=fixed_learning_rate, weight_decay=l2_reg_weight)
)

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
    signal_complex = signal_vals[:, 0] + 1j * signal_vals[:, 1]
    signal_trimmed = signal_complex[4 * zero_pad : -4 * zero_pad]
    x_trimmed = x_range[4 * zero_pad : -4 * zero_pad]
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
        integral_real = jnp.trapezoid(jnp.real(integrand), dx=DX)
        integral_imag = jnp.trapezoid(jnp.imag(integrand), dx=DX)
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

def reconstruct_complex(signal_vals):
    signal_vals = jnp.asarray(signal_vals)
    if signal_vals.ndim == 1 and signal_vals.shape[0] % 2 == 0:
        half = signal_vals.shape[0] // 2
        return signal_vals[:half] + 1j * signal_vals[half:]
    elif signal_vals.ndim == 2 and signal_vals.shape[1] == 2:
        return signal_vals[:, 0] + 1j * signal_vals[:, 1]
    else:
        raise ValueError("Unexpected shape for signal_vals:", signal_vals.shape)

def split_complex_to_channels(matrix):
    return np.stack([matrix.real, matrix.imag], axis=-1)

label_df = pd.read_csv(config['paths']['label_data_file_path'], dtype=str)
data_df = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str)
label_matrix = label_df.map(convert_to_complex).to_numpy().T
data_matrix = data_df.map(convert_to_complex).to_numpy().T

x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values

label_matrix_split = split_complex_to_channels(label_matrix)
data_matrix_split = split_complex_to_channels(data_matrix)
dataset = list(zip(data_matrix_split, label_matrix_split))

with open(config['paths']['setup_file_path']) as f:
    setup = json.load(f)

F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values

dx = 0.25
zero_pad = 50

test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading test dataset...")
    test_label_df = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
    test_data_df = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)
    test_label_matrix = test_label_df.map(convert_to_complex).to_numpy().T
    test_data_matrix = test_data_df.map(convert_to_complex).to_numpy().T
    test_label_matrix_split = split_complex_to_channels(test_label_matrix)
    test_data_matrix_split = split_complex_to_channels(test_data_matrix)
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

def loss_fn(params, model, x, y, deterministic, rng, fw, iw):
    preds = model.apply({'params': params}, x, mutable=False)

    if preds.shape != y.shape:
        preds = preds.reshape(y.shape)

    # === Fourier domain MSE (per-channel) ===
    real_mse = jnp.mean((preds[..., 0] - y[..., 0]) ** 2)
    imag_mse = jnp.mean((preds[..., 1] - y[..., 1]) ** 2)
    mse_loss = (real_mse + imag_mse) / 2.0

    # === Image domain loss ===
    def compute_image_norm(signal_sample, pred_coeffs, true_coeffs):
        preds_real = pred_coeffs[..., 0]
        preds_imag = pred_coeffs[..., 1]
        true_real = true_coeffs[..., 0]
        true_imag = true_coeffs[..., 1]

        return calculate_image_norm(
            x_range, signal_sample, preds_real, preds_imag,
            true_real, true_imag,
            kpsi_values, ionoNHarm, F, dx, xi, zero_pad
        )

    batch_size = x.shape[0]
    sample_indices = jax.random.choice(rng, batch_size, shape=(batch_size,), replace=False)

    image_loss = jnp.mean(jax.vmap(lambda i: compute_image_norm(
        x[i], preds[i], y[i]))(sample_indices))

    return fw * mse_loss + iw * image_loss



def train_step(state, model, x, y, rng, fw, iw):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, model, x, y, False, rng, fw, iw)

    # Compute gradient statistics
    def grad_stats(grad_tree):
        grads_flat, _ = jax.tree_util.tree_flatten(grad_tree)
        norms = jnp.array([jnp.linalg.norm(g) for g in grads_flat if g is not None])
        return norms.mean(), norms.std()

    grad_mean, grad_std = grad_stats(grads)
    print(f"Gradient mean: {grad_mean:.4e}, std: {grad_std:.4e}")
    state = state.apply_gradients(grads=grads)
    return state, loss

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1], 2)
variables = model.init(main_key, jnp.ones(input_shape))

if os.path.exists(cached_weights_path):
    print("Loading saved model weights...")
    with open(cached_weights_path, "rb") as f:
        loaded_params = pickle.load(f)
    state = train_state.TrainState.create(apply_fn=model.apply, params=loaded_params, tx=opt)
else:
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

loss_history = []
test_loss_history = []
with open("training_losses_conv.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    batch_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        state, loss = train_step(state, model, batch_signal, batch_coefficients, subkey, fourier_weight, image_diff_weight)
        batch_loss += loss

    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // config['training']['batch_size']
        for test_signal, test_coefficients in data_loader(test_dataset, config['training']['batch_size'], shuffle=False):
            test_loss += loss_fn(state.params, model, test_signal, test_coefficients, True, rng_key, fourier_weight, image_diff_weight)
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_conv.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])

    print(f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}, Test Loss = {avg_test_loss:.6f}" if avg_test_loss else f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}")

final_weights_name = f"model_weights_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")

with open(f"{final_weights_name}_config.yaml", "w") as f:
    yaml.dump(config, f)