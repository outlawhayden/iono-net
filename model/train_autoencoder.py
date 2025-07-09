import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
import yaml
import json
import pickle
import optax
import csv
from flax.training import train_state
from datetime import datetime
from tqdm import tqdm
from AutoencoderModel import HybridAutoencoder
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
jax.config.update("jax_enable_x64", True)

# === Load Config ===
with open("config_autoencoder.yaml", "r") as f:
    config = yaml.safe_load(f)

# === RNG Setup ===
root_key = jax.random.PRNGKey(config["seed"])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

# === Model Setup ===
activation_fn = getattr(jnp, config["model"]["activation"])

model = HybridAutoencoder(
    up_dims=config["model"]["up_dims"],
    dense_dims=config["model"]["dense_dims"],
    down_dims=config["model"]["down_dims"],
    activation_fn=activation_fn,
    dropout_rate=0.3,
)


# === Data Load Helpers ===
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    return complex(s.replace("i", "j"))

def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    normalized = matrix / amp_max
    return normalized.real + 1j * normalized.imag

def stack_complex_to_channel(complex_array):
    return np.stack([complex_array.real, complex_array.imag], axis=-1)

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

# === Load Data ===
label_df = pd.read_csv(config["paths"]["label_data_file_path"], dtype=str)
data_df = pd.read_csv(config["paths"]["signal_data_file_path"], dtype=str)
label_matrix = normalize_complex_to_unit_range(label_df.map(convert_to_complex).to_numpy().T)
data_matrix = normalize_complex_to_unit_range(data_df.map(convert_to_complex).to_numpy().T)
data_matrix_stacked = stack_complex_to_channel(data_matrix)
label_matrix_split = split_complex_to_imaginary(label_matrix)
dataset = list(zip(data_matrix_stacked, label_matrix_split))

x_range = pd.read_csv(config["paths"]["x_range_file_path"]).iloc[:, 0].values

with open(config["paths"]["setup_file_path"]) as f:
    setup = json.load(f)
F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
kpsi_values = pd.read_csv(config["paths"]["kpsi_file_path"]).values

zero_pad = 50
dx = 0.25

# === Optional Test Dataset ===
test_dataset = None
if "test_data_file_path" in config["paths"] and "test_label_file_path" in config["paths"]:
    print("Loading test dataset...")
    test_label_df = pd.read_csv(config["paths"]["test_label_file_path"], dtype=str)
    test_data_df = pd.read_csv(config["paths"]["test_data_file_path"], dtype=str)
    test_label_matrix = normalize_complex_to_unit_range(test_label_df.map(convert_to_complex).to_numpy().T)
    test_data_matrix = normalize_complex_to_unit_range(test_data_df.map(convert_to_complex).to_numpy().T)
    test_data_matrix_stacked = stack_complex_to_channel(test_data_matrix)
    test_label_matrix_split = split_complex_to_imaginary(test_label_matrix)
    test_dataset = list(zip(test_data_matrix_stacked, test_label_matrix_split))

# === Data Loader ===
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

# === L4 Loss ===
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
        return jnp.trapezoid(jnp.abs(integrand)**4, dx=DX) / F

    image_vals = jax.vmap(evaluate_single)(x_trimmed)
    return jnp.sum(image_vals)

# === Loss Function ===
def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    sq_diffs = (preds_real - true_real) ** 2 + (preds_imag - true_imag) ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6)**2) * sq_diffs, axis=1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6)**4) * sq_diffs, axis=1))
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    if add_l4:
        sample_indices = jax.random.choice(rng_key, inputs.shape[0], shape=(4,), replace=False)
        l4_losses = []
        for i in sample_indices:
            loss = calculate_l4_norm(
                x_range, inputs[i], preds_real[i], preds_imag[i],
                kpsi_values, ionoNHarm, F, dx, xi, zero_pad
            )
            l4_losses.append(loss)
        loss_l4 = jnp.mean(jnp.stack(l4_losses))
    else:
        loss_l4 = 0.0

    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4)

# === Optimizer Setup ===
opt = optax.chain(
    optax.clip_by_global_norm(config['training']['gradient_clip_value']),
    optax.adamw(config["learning_rate"]["fixed"], weight_decay=config["training"]["l2_reg_weight"])
)

input_shape = (config["training"]["batch_size"], data_matrix_stacked.shape[1], 2)
variables = model.init(main_key, jnp.ones(input_shape), deterministic=True)
params = variables["params"]
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

# === Training ===
loss_history = []
test_loss_history = []
with open("training_losses_full.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config["optimizer"]["maxiter_adam"]), desc="Training"):
    batch_loss = 0.0
    num_batches = len(dataset) // config["training"]["batch_size"]
    for batch_signal, batch_coeffs in data_loader(dataset, config["training"]["batch_size"]):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coeffs,
            deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
            l2_reg_weight=config["training"]["l2_reg_weight"],
            fourier_weight=config["training"]["fourier_weight"],
            fourier_d1_weight=config["training"]["fourier_d1_weight"],
            fourier_d2_weight=config["training"]["fourier_d2_weight"],
            l4_weight=config["training"]["l4_reg_weight"]
        )
        state = state.apply_gradients(grads=grads)
        batch_loss += loss

    avg_train_loss = batch_loss / num_batches
    loss_history.append(avg_train_loss.item())

    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // config["training"]["batch_size"]
        for test_signal, test_coeffs in data_loader(test_dataset, config["training"]["batch_size"], shuffle=False):
            test_loss += loss_fn(
                state.params, model, test_signal, test_coeffs,
                deterministic=True, rng_key=rng_key,
                ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=False,
                l2_reg_weight=config["training"]["l2_reg_weight"],
                fourier_weight=config["training"]["fourier_weight"],
                fourier_d1_weight=config["training"]["fourier_d1_weight"],
                fourier_d2_weight=config["training"]["fourier_d2_weight"],
                l4_weight=config["training"]["l4_reg_weight"]
            )
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_full.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_test_loss])

    if (epoch + 1) % 100 == 0:
        fname = f"model_weights_epoch_{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(state.params, f)
        print(f"Saved model weights to {fname}")

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}" + (f", Test Loss = {avg_test_loss:.6f}" if avg_test_loss else ""))

# === Final Save ===
final_fname = f"model_weights_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_fname, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Final weights saved to {final_fname}")
