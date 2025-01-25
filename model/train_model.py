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
import csv

# List available GPU devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")

# Load configurations
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set GPU device for training
gpu_id = config.get("gpu_id", 0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print(f"Training on GPU {gpu_id}")

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

label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)
label_matrix = label_df.map(convert_to_complex).to_numpy().T
data_matrix = data_df.map(convert_to_complex).to_numpy().T

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

label_matrix_split = split_complex_to_imaginary(label_matrix)
data_matrix_split = split_complex_to_imaginary(data_matrix)

dataset = list(zip(data_matrix_split, label_matrix_split))

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

def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    loss_real = jnp.mean((preds_real - true_real) ** 2)
    loss_imag = jnp.mean((preds_imag - true_imag) ** 2)
    return loss_real + loss_imag

opt = optax.adam(optax.exponential_decay(
    config['learning_rate']['initial'],
    config['learning_rate']['step'],
    config['learning_rate']['gamma'],
    end_value=config['learning_rate']['final']
))

state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

loss_history = []

# Training loop
for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
    batch_loss = 0.0
    num_batches = len(dataset) // config['training']['batch_size']
    for batch_signal, batch_coefficients in data_loader(dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey
        )
        state = state.apply_gradients(grads=grads)
        batch_loss += loss
    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())
    #print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

# Save model weights
datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
weights_filename = f'model_weights_{datestr}.pkl'
with open(weights_filename, 'wb') as weights_file:
    pickle.dump(state.params, weights_file)
print(f"Model weights saved to {weights_filename}")

# Export loss history to CSV
loss_history_csv_filename = f'loss_history_{datestr}.csv'
with open(loss_history_csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Loss"])
    for i, loss in enumerate(loss_history):
        csvwriter.writerow([i + 1, loss])

print(f"Loss history saved to {loss_history_csv_filename}")
