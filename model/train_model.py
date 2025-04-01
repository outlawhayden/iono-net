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
from Helper import *
from Image import *
from Psi import *
from Optimize import *


os.environ["JAX_TRACEBACK_FILTERING"] = "off"
cached_weights_path = "model_weights.pkl"


# List available GPU devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")

# Enable multi-GPU training
if num_gpus > 1:
    print("Using multiple GPUs for training.")
    jax.config.update("jax_platform_name", "gpu")



# Load configurations
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize JAX keys
root_key = jax.random.PRNGKey(seed=config['seed'])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

# Instantiate the model
architecture = config['model']['architecture']
activation_fn = getattr(jnp, config['model']['activation'])
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

# Load data
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace('i', 'j'))

label_file_path = config['paths']['label_data_file_path']
data_file_path = config['paths']['signal_data_file_path']
x_range_file_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_values_path = config['paths']['kpsi_file_path']

label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)
label_matrix = label_df.map(convert_to_complex).to_numpy().T
data_matrix = data_df.map(convert_to_complex).to_numpy().T
x_range = pd.read_csv(x_range_file_path).iloc[:, 0].values

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

label_matrix_split = split_complex_to_imaginary(label_matrix)
data_matrix_split = split_complex_to_imaginary(data_matrix)

dataset = list(zip(data_matrix_split, label_matrix_split))

with open(setup_path) as f:
    setup = json.load(f)

F, ionoNHarm, xi, windowType, sumType = (
    setup["F"], 
    setup["ionoNharm"], 
    setup["xi"], 
    setup["windowType"], 
    setup["sumType"],
)

kpsi_values = pd.read_csv(kpsi_values_path).values

dx = 0.25  # hard coded for now
zero_pad = 50  # padding of zeros on each side of signal, hard coded for now

# Load test dataset if available
test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading test dataset...")
    test_label_file_path = config['paths']['test_label_file_path']
    test_data_file_path = config['paths']['test_data_file_path']
    
    test_label_df = pd.read_csv(test_label_file_path, dtype=str)
    test_data_df = pd.read_csv(test_data_file_path, dtype=str)
    test_label_matrix = test_label_df.map(convert_to_complex).to_numpy().T
    test_data_matrix = test_data_df.map(convert_to_complex).to_numpy().T
    
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

input_shape = (config['training']['batch_size'], data_matrix_split.shape[1])
variables = model.init(root_key, jnp.ones(input_shape), deterministic=True)

# L2 regularization weight
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
l4_reg_weight = config['training'].get('l4_reg_weight', 1e-3)
fourier_weight = config['training'].get('fourier_weight', 1e-3)
fourier_d1_weight = config['training'].get('fourier_d1_weight', 1e-3)
fourier_d2_weight = config['training'].get('fourier_d2_weight', 1e-3)

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



def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key, ionoNHarm, kpsi_values, add_l4):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    real_diffs = preds_real - true_real
    imag_diffs = preds_imag - true_imag
    
    direct_loss = jnp.sqrt(jnp.sum(jnp.array([(real_diffs[:, i]**2 + imag_diffs[:, i]**2) for i in range(6)])))
    d1_loss = jnp.sqrt(jnp.sum(jnp.array([(i ** 2) *(real_diffs[:, i]**2 + imag_diffs[:, i]**2) for i in range(6)])))
    d2_loss = jnp.sqrt(jnp.sum(jnp.array([(i ** 4) *(real_diffs[:, i]**2 + imag_diffs[:, i]**2) for i in range(6)])))



    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    if add_l4:
        # Select random sample from batch for L4 loss
        batch_size = inputs.shape[0]
        batch_size = inputs.shape[0]
        num_l4_samples = 4  # ← pick however many you want to average over
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
    return fourier_weight * direct_loss + fourier_d1_weight * d1_loss + fourier_d2_weight * d2_loss + l2_reg_weight * l2_loss + l4_reg_weight * loss_l4

gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)

opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(
        learning_rate=optax.exponential_decay(
            config['learning_rate']['initial'],
            config['learning_rate']['step'],
            config['learning_rate']['gamma'],
            end_value=config['learning_rate']['final']
        ),
        weight_decay=l2_reg_weight
    )
)

if os.path.exists(cached_weights_path):
    print("Loading saved model weights...")
    with open(cached_weights_path, "rb") as f:
        loaded_params = pickle.load(f)
    state = train_state.TrainState.create(apply_fn=model.apply, params=loaded_params, tx=opt)
    print("Model weights loaded successfully.")
else:
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)
    print("No saved weights found. Training from scratch.")
loss_history = []
test_loss_history = []

# Create CSV file and write headers
with open("training_losses.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    batch_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']
    
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey,
            ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4 = True
        )
        state = state.apply_gradients(grads=grads)
        batch_loss += loss
    
    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())
    
    # Evaluate test loss if test dataset is available
    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // config['training']['batch_size']
        for test_signal, test_coefficients in data_loader(test_dataset, config['training']['batch_size'], shuffle=False):
            test_loss += loss_fn(state.params, model, test_signal, test_coefficients, deterministic=True, rng_key=rng_key, ionoNHarm=ionoNHarm, kpsi_values=kpsi_values, add_l4 = True)
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    # Append epoch results to the CSV file
    with open("training_losses.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])
    
    print(f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}, Test Loss = {avg_test_loss:.6f}" if avg_test_loss else f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}")

# Save model weights to a pickle file
with open("model_weights.pkl", "wb") as f:
    pickle.dump(state.params, f)
 
print("Training complete. Model weights saved as 'model_weights.pkl'.")
