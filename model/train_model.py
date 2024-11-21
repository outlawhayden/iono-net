import jax.numpy as jnp
import os
import jax
from model import ConfigurableModel 
import yaml
from jax import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import enlighten
import pickle
from datetime import datetime

from flax.training import train_state
import optax

# Load configurations from YAML file
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devices = jax.local_devices()

print("Backend Selected:", jax.lib.xla_bridge.get_backend().platform)
print("Detected Devices:", jax.devices())

root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

# Load model hyperparameters
learning_rate = config['learning_rate']
model_params = config['model']
batch_size = config['training']['batch_size']
max_epochs = config['optimizer']['maxiter_adam']

# Instantiate model with config parameters
architecture = model_params['trunk_architecture']
activation_fn = getattr(jnp, model_params['trunk_activation'])
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

# Define the path to the CSV files
label_file_path = '/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/compl_ampls_20241104_212602.csv'
data_file_path = '/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/uscStruct_vals_20241104_212602.csv'
# Function to convert complex strings (e.g., '5.7618732844527+1.82124094798357i') to complex numbers
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace('i', 'j'))

# Load the CSV files using pandas and apply conversion to complex numbers
label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)

label_df = label_df.dropna(axis = 1, how = 'any')
data_df = data_df.dropna(axis = 1, how = 'any')

# Convert the string representations into complex values
label_matrix = label_df.applymap(convert_to_complex).to_numpy().T  # Transpose to get data points as rows
data_matrix = data_df.applymap(convert_to_complex).to_numpy().T    # Transpose to get data points as rows

# Split complex matrices into real and imaginary parts
def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

# Now each row represents a data point with real and imaginary parts concatenated along the row
label_matrix_split = split_complex_to_imaginary(label_matrix)
data_matrix_split = split_complex_to_imaginary(data_matrix)

print("Label Matrix Split Shape:", label_matrix_split.shape)
print("Data Matrix Split Shape:", data_matrix_split.shape)

# Combine the signal (data) and coefficients (labels) into a dataset
dataset = list(zip(data_matrix_split, label_matrix_split))

# Data loader function
def data_loader(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    
    # Shuffle dataset if required
    if shuffle:
        np.random.shuffle(indices)
    
    # Loop over dataset and yield batches
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        
        # Extract the batch of signals and coefficients separately
        batch_signal = [dataset[i][0] for i in batch_indices]  # Signal of length 2882
        batch_coefficients = [dataset[i][1] for i in batch_indices]  # Coefficients of length 12
        
        # Convert the batch data to JAX arrays
        yield jnp.array(batch_signal), jnp.array(batch_coefficients)

# Adjust input shape based on actual data dimensions
input_shape = (batch_size, data_matrix_split.shape[1])  # Shape based on the actual data

# Initialize model variables with correct input shape
variables = model.init(root_key, jnp.ones(input_shape), deterministic=True)

# Define the loss function and optimizer
def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    loss_real = jnp.mean((preds_real - true_real) ** 2)
    loss_imag = jnp.mean((preds_imag - true_imag) ** 2)
    return loss_real + loss_imag

opt = optax.adam(optax.exponential_decay(
    learning_rate['initial'],
    learning_rate['step'],
    learning_rate['gamma'],
    end_value=learning_rate['final']
))

class TrainState(train_state.TrainState):
    loss_fn = staticmethod(loss_fn)

state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

log_loss = []  # Track the loss value for each epoch
log_minloss = []  # Track the minimum loss value observed so far
minloss = float('inf')  # Initialize minloss with a high value

# Progress bar setup using enlighten for visual feedback during training
manager = enlighten.get_manager()
pbar_outer = manager.counter(
    total=max_epochs, 
    desc="Training", 
    unit="epochs", 
    color="red", 
    bar_format='{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s] Current Loss: {current_loss:1.6f} Best Loss: {best_loss:1.6f}'
)

# Main training loop
for epoch in range(max_epochs):
    batch_loss = 0.0  # Track cumulative loss over the batches in each epoch
    num_batches = len(dataset) // batch_size
    
    for batch_signal, batch_coefficients in data_loader(dataset, batch_size):
        rng_key, subkey = jax.random.split(rng_key)

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(state.loss_fn)(
            state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey
        )
        
        # Apply gradients to update model parameters
        state = state.apply_gradients(grads=grads)
        
        # Accumulate batch loss
        batch_loss += loss

    avg_epoch_loss = batch_loss / num_batches

    # Update the best (minimum) loss if the current one is better
    if avg_epoch_loss < minloss:
        minloss = avg_epoch_loss
        params_opt = state.params  # Save the parameters with the lowest loss

    # Update the progress bar with the current average loss and best loss
    pbar_outer.update(current_loss=avg_epoch_loss, best_loss=minloss, increment=1)
    
    # Log the loss for this epoch
    log_loss.append(avg_epoch_loss)
    log_minloss.append(minloss)

# Stop the progress bar once training is complete
manager.stop()
print(f"Training completed. Final loss: {log_loss[-1]}, Minimum loss: {minloss}")

# Export optimal parameters to pickle
datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'model_params_{datestr}.pkl', 'wb') as f:
    pickle.dump(params_opt, f)
