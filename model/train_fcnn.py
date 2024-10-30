import flax.linen as nn
import jax.numpy as jnp
import os
import jax
from jax import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import enlighten
import pickle
from datetime import datetime

from flax.training import train_state
import optax

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devices = jax.local_devices()

print("Backend Selected:", jax.lib.xla_bridge.get_backend().platform)
print("Detected Devices:", jax.devices())



root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)



# training config dict
tr_config = {
    "lr_0" : 0.003,              # Initial learning rate for the optimizer
    "lr_gamma": 0.95,            # Learning rate decay factor (multiplies the learning rate by this value at each step)
    "lr_step" : 2000,            # Number of iterations after which the learning rate is updated (decayed)
    "lr_f" : 1e-5,               # Final learning rate (smallest allowed learning rate after decay)
    "maxiter_adam" : 20000,       # Maximum number of iterations for the Adam optimizer
    "maxiter_lbfgs": 1000,       # Maximum number of iterations for the L-BFGS optimizer
    "deepOnet_width" : 12,       # Width of the hidden layers in the DeepONet model (number of neurons per layer)
    "trunk_architecture" : [50, 30, 10],  # Architecture of the trunk network, with 3 layers each containing 50 neurons
    "trunk_activation": jnp.tanh, # Activation function used in the trunk network (tanh in this case)
    "trunk_input_dim": 1,        # Input dimension for the trunk network
    "trunk_output_dim": 1,       # Output dimension for the trunk network
    "trunk_sensor": 2881,         # Number of sensors (inputs) for the trunk network
    "num_train": 1000,             # Number of training samples
    "num_test": 300                # Number of test samples
}



###
batch_size = 100
###




# Define the path to the CSV files
label_file_path = '/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/compl_ampls_20241029_193914.csv'
data_file_path = '/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/nuStruct_withSpeckle_20241029_193914.csv'

# Function to convert complex strings (e.g., '5.7618732844527+1.82124094798357i') to complex numbers
def convert_to_complex(s):
    return complex(s.replace('i', 'j'))

# Load the CSV files using pandas and apply conversion to complex numbers
label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)

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
# Each signal now has length 2000 (real + imaginary), coefficients have length 12 (real + imaginary)
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
        batch_signal = [dataset[i][0] for i in batch_indices]  # Signal of length 2000
        batch_coefficients = [dataset[i][1] for i in batch_indices]  # Coefficients of length 12
        
        # Convert the batch data to JAX arrays
        yield jnp.array(batch_signal), jnp.array(batch_coefficients)



num_batches = len(dataset) // batch_size


class ComplexFCNN(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic, rngs={'dropout': None}):  # Remove rng_key as a default here
        # First dense layer: input shape should match the length of the signal (2000)
        
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        x = nn.Dense(128)(x)  # First fully connected layer
        x = nn.relu(x)
        
        # Apply dropout after the first layer
        x = nn.Dropout(0.2)(x, deterministic=deterministic)
        
        # Second dense layer
        x = nn.Dense(64)(x)   # Second fully connected layer
        x = nn.relu(x)
        
        # Final dense layer to output 12 values (6 real and 6 imaginary values for coefficients)
        x = nn.Dense(12)(x)  # Output layer with 12 units
        return x

# Instance of the model
model = ComplexFCNN()

# Set seed
key = jax.random.PRNGKey(0)

signal_length = 1441

input_shape = (batch_size, 2 * signal_length)

# Initialize model variables with random inputs of ones (for testing initialization)
variables = model.init(key, jnp.ones(input_shape), deterministic=True)

# Forward pass to test the configuration and output shape
output = model.apply(variables, jnp.ones(input_shape), deterministic=True)


def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
    # Forward pass with the deterministic flag and PRNG key for dropout
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    
    # Split predictions into real and imaginary parts
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]

    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    
    # MSE loss for real and imaginary parts
    loss_real = jnp.mean((preds_real - true_real) ** 2)
    loss_imag = jnp.mean((preds_imag - true_imag) ** 2)
    
    return loss_real + loss_imag


# Training configuration parameters
maxepochs  = tr_config["maxiter_adam"]  # Maximum number of epochs for training with Adam optimizer
lr0        = tr_config["lr_0"]          # Initial learning rate for Adam optimizer
decay_rate = tr_config["lr_gamma"]      # Learning rate decay factor
decay_step = tr_config["lr_step"]       # Number of steps after which the learning rate is decayed
lrf        = tr_config["lr_f"]          # Final learning rate (minimum after decay)

# Progress bar setup using enlighten for visual feedback during training
manager = enlighten.get_manager()  # Initialize the progress manager
outer_bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                 u'{count:{len_total}d}/{total:d} ' + \
                 u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s] ' + \
                 u'{loss}{current_loss:1.4e}'  # Custom format for the progress bar, showing loss and time details

# Create the progress bar for the outer loop (training epochs)
pbar_outer = manager.counter(
    total=maxepochs,               # Total number of epochs (training iterations)
    desc="Main Loop",              # Description of the progress bar
    unit="epochs",                 # Unit to track (epochs)
    color="red",                   # Color of the progress bar
    loss="loss=",                  # Prefix for displaying the loss value
    current_loss=1e+9,             # Initial high loss value for display
    bar_format=outer_bar_format    # Format defined above for the progress bar
)

# Optimizer setup using Optax (Adam optimizer with exponential learning rate decay)
# Learning rate decays over time with the given decay rate and final value
opt_adam = optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, end_value=lrf))

# Initialize training state
class TrainState(train_state.TrainState):
    loss_fn = staticmethod(loss_fn)

tx = opt_adam
state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

# Initialize lists to log loss values and minimum loss over epochs
log_loss  = []  # Track the loss value for each epoch
log_minloss = []  # Track the minimum loss value observed so far

# Main training loop for the specified number of epochs
for epoch in range(maxepochs):
    batch_loss = 0.0  # Track cumulative loss over the batches in each epoch
    num_batches = len(dataset) // batch_size  # Assuming dataset length is known
    
    for batch_signal, batch_coefficients in data_loader(dataset, batch_size):
        # Generate a new PRNG key for dropout for this batch
        rng_key, subkey = jax.random.split(rng_key)

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(state.loss_fn)(
            state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey
        )
        
        # Apply gradients to update model parameters
        state = state.apply_gradients(grads=grads)
        
        # Accumulate batch loss
        batch_loss += loss

    # Average batch loss for the epoch
    avg_epoch_loss = batch_loss / num_batches

    # Update the progress bar with the current average loss
    pbar_outer.update(current_loss=avg_epoch_loss, increment=1)
    
    # Log the loss for this epoch
    log_loss.append(avg_epoch_loss)
    
    # Initialize or update the minimum loss and save the parameters
    if epoch == 0 or avg_epoch_loss < minloss:
        minloss = avg_epoch_loss
        params_opt = state.params  # Save the parameters with the lowest loss
    
    log_minloss.append(minloss)

# Stop the progress bar once training is complete
manager.stop()

# Optionally, print or save the logs
print(f"Training completed. Final loss: {log_loss[-1]}, Minimum loss: {minloss}")

# Export optimal parameters to pickle
datestr = datetime.now().strftime('%Y%m%d_%H%M%S')

with open(f'model_params_{datestr}.pkl', 'wb') as f:
    pickle.dump(params_opt, f)