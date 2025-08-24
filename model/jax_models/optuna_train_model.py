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
import optuna
import time as time_module
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights.pkl"

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
activation_fn = getattr(jnp, config['model']['activation'])

def apply_fft_to_signals(matrix):
    return np.fft.fft(matrix, axis=1)

def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace('i', 'j'))
    
def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1  # Prevent division by zero
    normalized = matrix / amp_max
    return normalized.real + 1j * normalized.imag



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

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return 2 * (matrix - min_val) / (max_val - min_val) - 1


# Disable FFT for now
data_matrix_fft = data_matrix
data_matrix_split = split_complex_to_imaginary(data_matrix_fft)
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


def calculate_l4_norm(x_range, signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    # Trim and convert signal
    signal_complex = signal_vals[:1441] + 1j * signal_vals[1441:]
    signal_trimmed = signal_complex[4 * zero_pad : -4 * zero_pad]
    x_trimmed = x_range[4 * zero_pad : -4 * zero_pad]

    # Set up base domain for evaluation
    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

    def evaluate_single(y):
        base = y + offsets

        # Interpolate real/imag parts of signal
        real_interp = jnp.interp(base, x_trimmed, jnp.real(signal_trimmed))
        imag_interp = jnp.interp(base, x_trimmed, jnp.imag(signal_trimmed))
        signal_interp = real_interp + 1j * imag_interp

        # Uniform window
        window = jnp.ones_like(base)

        # SAR waveform
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)

        # Phase modulation (psi)
        sarr = xi * base + (1 - xi) * y
        psi_cos = jnp.sum(preds_real[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_sin = jnp.sum(preds_imag[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_vals = jnp.exp(1j * (psi_cos + psi_sin))

        integrand = waveform * signal_interp * window * psi_vals
        integral_real = jnp.trapezoid(jnp.real(integrand), dx=DX)
        integral_imag = jnp.trapezoid(jnp.imag(integrand), dx=DX)

        return (integral_real + 1j * integral_imag) / F

    image_vals = vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4)

def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]

    # Elementwise squared diffs
    real_diffs = preds_real - true_real
    imag_diffs = preds_imag - true_imag
    sq_diffs = real_diffs ** 2 + imag_diffs ** 2

    # Mean-squared losses
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 2) * sq_diffs, axis=1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 4) * sq_diffs, axis=1))

    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    # L4 loss over sampled batch entries
    if add_l4:
        batch_size = inputs.shape[0]
        num_l4_samples = 4
        sample_indices = jax.random.choice(rng_key, batch_size, shape=(num_l4_samples,), replace=False)

        def compute_single_l4(index):
            signal_sample = inputs[index]
            preds_real_sample = preds_real[index]
            preds_imag_sample = preds_imag[index]
            return calculate_l4_norm(
                x_range,
                signal_sample,
                preds_real_sample,
                preds_imag_sample,
                kpsi_values,
                ionoNHarm,
                F,
                dx,
                xi,
                zero_pad
            )

        loss_l4 = jnp.mean(jax.vmap(compute_single_l4)(sample_indices))
    else:
        loss_l4 = 0.0

    # Combine all weighted loss terms
    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4)



def create_optimizer(trial, l2_reg_weight):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    clip_value = trial.suggest_float("gradient_clip_value", 0.1, 5.0)
    opt = optax.chain(
        optax.clip_by_global_norm(clip_value),
        optax.adamw(learning_rate=lr, weight_decay=l2_reg_weight)
    )
    return opt, lr, clip_value

def objective(trial):
    global rng_key
    print(f"Starting trial {trial.number}...")
    start_time = time_module.time()

    # Dynamic architecture
    num_layers = trial.suggest_int("num_layers", 1, 6)
    architecture = [trial.suggest_categorical(f"layer_{i}", [16, 32, 64, 128, 256]) for i in range(num_layers)]
    model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

    # Loss weights (normalized)
    fw = trial.suggest_float("fourier_weight", 1e-5, 1.0)
    d1w = trial.suggest_float("fourier_d1_weight", 1e-5, 1.0)
    d2w = trial.suggest_float("fourier_d2_weight", 1e-5, 1.0)
    total = fw + d1w + d2w + 1e-8
    fourier_weight = fw / total
    fourier_d1_weight = d1w / total
    fourier_d2_weight = d2w / total

    # Other hyperparameters
    l2_reg_weight = trial.suggest_float("l2_reg_weight", 1e-6, 1e-2, log=True)
    l4_weight = trial.suggest_float("l4_weight", 1e-5, 1.0)
    opt, lr, clip_val = create_optimizer(trial, l2_reg_weight)

    input_shape = (config['training']['batch_size'], data_matrix_split.shape[1])
    variables = model.init(main_key, jnp.ones(input_shape), deterministic=True)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

    best_val_loss = float("inf")
    for epoch in range(120):
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

        if test_dataset:
            test_loss = 0.0
            test_batches = len(test_dataset) // config['training']['batch_size']
            for test_signal, test_coefficients in data_loader(test_dataset, config['training']['batch_size'], shuffle=False):
                test_loss += loss_fn(
                    state.params, model, batch_signal, batch_coefficients,
                    deterministic=False, rng_key=subkey,
                    ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4=True,
                    l2_reg_weight=l2_reg_weight,
                    fourier_weight=fourier_weight,
                    fourier_d1_weight=fourier_d1_weight,
                    fourier_d2_weight=fourier_d2_weight,
                    l4_weight=l4_weight
                )
            avg_test_loss = test_loss / test_batches
            best_val_loss = min(best_val_loss, avg_test_loss.item())

    elapsed_time = time_module.time() - start_time
    print(f"Trial {trial.number} completed in {elapsed_time:.2f} seconds. Loss: {best_val_loss:.6f}")
    return best_val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
