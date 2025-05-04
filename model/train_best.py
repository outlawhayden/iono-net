import os
import jax
import jax.numpy as jnp
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
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights.pkl"

with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

root_key = jax.random.PRNGKey(seed=config['seed'])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)
activation_fn = getattr(jnp, config['model']['activation'])

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

label_df = pd.read_csv(config['paths']['label_data_file_path'], dtype=str)
data_df = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str)
label_matrix = normalize_complex_to_unit_range(label_df.map(convert_to_complex).to_numpy().T)
data_matrix = normalize_complex_to_unit_range(data_df.map(convert_to_complex).to_numpy().T)
data_matrix_split = split_complex_to_imaginary(data_matrix)
label_matrix_split = split_complex_to_imaginary(label_matrix)
dataset = list(zip(data_matrix_split, label_matrix_split))

# Load test data
label_df_test = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
data_df_test = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)
label_matrix_test = normalize_complex_to_unit_range(label_df_test.map(convert_to_complex).to_numpy().T)
data_matrix_test = normalize_complex_to_unit_range(data_df_test.map(convert_to_complex).to_numpy().T)
data_matrix_split_test = split_complex_to_imaginary(data_matrix_test)
label_matrix_split_test = split_complex_to_imaginary(label_matrix_test)
test_dataset = list(zip(data_matrix_split_test, label_matrix_split_test))

x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values

with open(config['paths']['setup_file_path']) as f:
    setup = json.load(f)
F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
dx = 0.25
zero_pad = 50

def calculate_l4_norm(x_range, signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    signal_complex = signal_vals[:1441] + 1j * signal_vals[1441:]
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
        integral_real = jnp.trapezoid(jnp.real(integrand), dx=DX)
        integral_imag = jnp.trapezoid(jnp.imag(integrand), dx=DX)
        return (integral_real + 1j * integral_imag) / F

    image_vals = jax.vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4)

def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    sq_diffs = (preds_real - true_real) ** 2 + (preds_imag - true_imag) ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 2) * sq_diffs, axis=1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 4) * sq_diffs, axis=1))
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    if add_l4:
        batch_size = inputs.shape[0]
        sample_indices = jax.random.choice(rng_key, batch_size, shape=(4,), replace=False)

        def compute_single_l4(index):
            return calculate_l4_norm(x_range, inputs[index], preds_real[index], preds_imag[index],
                                     kpsi_values, ionoNHarm, F, dx, xi, zero_pad)

        loss_l4 = jnp.mean(jax.vmap(compute_single_l4)(sample_indices))
    else:
        loss_l4 = 0.0

    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4)

best_params = {'num_layers': 5, 'layer_0': 256, 'layer_1': 256, 'layer_2': 32, 'layer_3': 256, 'layer_4': 16, 
               'fourier_weight': 0.622799456429225, 'fourier_d1_weight': 0.41743259106954167, 'fourier_d2_weight': 0.9118997302542735, 'l2_reg_weight': 6.273046585460742e-06, 'l4_weight': 0.7378929122485713, 'learning_rate': 0.001465006379177463, 'gradient_clip_value': 3.945257782871415}

architecture = [best_params[f"layer_{i}"] for i in range(best_params["num_layers"])]
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

fw, d1w, d2w = best_params["fourier_weight"], best_params["fourier_d1_weight"], best_params["fourier_d2_weight"]
total = fw + d1w + d2w + 1e-8
fourier_weight = fw / total
fourier_d1_weight = d1w / total
fourier_d2_weight = d2w / total

opt = optax.chain(
    optax.clip_by_global_norm(best_params["gradient_clip_value"]),
    optax.adamw(learning_rate=best_params["learning_rate"], weight_decay=best_params["l2_reg_weight"])
)

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1])
variables = model.init(main_key, jnp.ones(input_shape), deterministic=True)
state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    print(f"Epoch {epoch + 1}/{config['optimizer']['maxiter_adam']}")
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coefficients,
            deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
            l2_reg_weight=best_params["l2_reg_weight"],
            fourier_weight=fourier_weight,
            fourier_d1_weight=fourier_d1_weight,
            fourier_d2_weight=fourier_d2_weight,
            l4_weight=best_params["l4_weight"]
        )
        state = state.apply_gradients(grads=grads)

with open("model_weights_best.pkl", "wb") as f:
    pickle.dump(state.params, f)

print("Training complete. Weights saved to model_weights_best.pkl")

# === Evaluate losses ===
def compute_full_loss(dataset, name="Set"):
    total_loss = 0
    count = 0
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size'], shuffle=False):
        loss = loss_fn(
            state.params, model, batch_signal, batch_coefficients,
            deterministic=True, rng_key=rng_key,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
            l2_reg_weight=best_params["l2_reg_weight"],
            fourier_weight=fourier_weight,
            fourier_d1_weight=fourier_d1_weight,
            fourier_d2_weight=fourier_d2_weight,
            l4_weight=best_params["l4_weight"]
        )
        total_loss += float(loss) * len(batch_signal)
        count += len(batch_signal)
    avg_loss = total_loss / count
    print(f"{name} loss: {avg_loss:.6f}")
    return avg_loss

train_loss = compute_full_loss(dataset, name="Train")
test_loss = compute_full_loss(test_dataset, name="Test")

# === Save losses ===
results_path = config['paths'].get("results_csv_path", "results.csv")
csv_headers = ["timestamp", "train_loss", "test_loss"]
row = [datetime.now().isoformat(), train_loss, test_loss]

if not os.path.exists(results_path):
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerow(row)
else:
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

print(f"Logged train and test loss to {results_path}")
