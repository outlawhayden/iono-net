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
from UNet1D import *
from tqdm import tqdm
import pickle
import json
import csv
from Helper import *
from Image import *
from Psi import *
from Optimize import *

# enable x64 precision for integration
jax.config.update("jax_enable_x64", True)

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# get num gpus
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")
if num_gpus > 1:
    print("Using multiple GPUs for training.")
    jax.config.update("jax_platform_name", "gpu")

# open config file
with open("/home/houtlaw/iono-net/model/config_unet.yaml", "r") as f:
    config = yaml.safe_load(f)

# set rng seeds
root_key = jax.random.PRNGKey(seed=config['seed'])

seed = config['seed']
np.random.seed(seed)
random.seed(seed)
root_key = jax.random.PRNGKey(seed)  
main_key, params_key, rng_key = jax.random.split(root_key, num=3)

# load config file paths
label_file_path = config['paths']['label_data_file_path']
data_file_path = config['paths']['data_file_path']
x_range_file_path = config['paths']['x_range_file_path']
setup_path = config['paths']['setup_file_path']
kpsi_values_path = config['paths']['kpsi_file_path']

# get misc info from setup file for dataset
with open(setup_path) as f:
    setup = json.load(f)

F, ionoNHarm, xi, windowType, sumType = setup["F"], setup["ionoNharm"], setup["xi"], setup["windowType"], setup["sumType"]
kpsi_values = pd.read_csv(kpsi_values_path).values

dx = 0.25
zero_pad = 50

def convert_to_complex(s): 
    if s == "NaNNaNi":
        return np.nan
    else:
        return complex(s.replace('i', 'j'))


def stack_real_imag_as_channels(complex_array):
    real_part = complex_array.real[..., np.newaxis]
    imag_part = complex_array.imag[..., np.newaxis]
    return np.concatenate([real_part, imag_part], axis=-1)

# load in data according to file paths
label_matrix_raw = pd.read_csv(label_file_path).map(convert_to_complex).to_numpy().T
data_matrix_raw = pd.read_csv(data_file_path).map(convert_to_complex).to_numpy().T
x_range = pd.read_csv(x_range_file_path).iloc[:,0].values #1040 long

# normalize and stack input data
data_matrix_real = data_matrix_raw.real
data_matrix_imag = data_matrix_raw.imag
data_mean = np.mean(data_matrix_raw)
data_std = np.std(data_matrix_raw)
data_matrix_norm = (data_matrix_raw - data_mean) / data_std
data_matrix = stack_real_imag_as_channels(data_matrix_norm.T)  # shape: (n_samples, signal_length, 2)

# normalize and stack label data
label_real = label_matrix_raw.real
label_imag = label_matrix_raw.imag
label_mean = np.mean(label_matrix_raw)
label_std = np.std(label_matrix_raw)
label_matrix_norm = (label_matrix_raw - label_mean) / label_std
label_matrix = stack_real_imag_as_channels(label_matrix_norm)  # shape: (n_samples, n_coeffs, 2)

# load test dataset if it exists
test_dataset = None
if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
    print("Loading Test Dataset")
    test_label_matrix_raw = pd.read_csv(config["paths"]["test_label_file_path"]).map(convert_to_complex).to_numpy().T
    test_label_matrix_norm = (test_label_matrix_raw - label_mean) / label_std
    test_label_matrix = stack_real_imag_as_channels(test_label_matrix_norm)

    test_data_matrix_raw = pd.read_csv(config["paths"]["test_data_file_path"]).map(convert_to_complex).to_numpy().T
    test_data_matrix_norm = (test_data_matrix_raw - data_mean) / data_std
    test_data_matrix = stack_real_imag_as_channels(test_data_matrix_norm.T)

    test_dataset = list(zip(test_data_matrix, test_label_matrix))
else:
    print("No Test Dataset Loaded")

# create zipped dataset
dataset = list(zip(data_matrix, label_matrix))

# load model architecture, instantiate
model_config = config["model_config"]
batch_size = config["training"]["batch_size"]

model = UNet1D(
    down_channels=model_config["down_channels"],
    bottleneck_channels=model_config["bottleneck_channels"],
    up_channels=model_config["up_channels"],
    output_dim=model_config["output_dim"]
)

signal_length = data_matrix.shape[1]  # already includes full length
x_dummy = jnp.ones((batch_size, signal_length, 2))

# Initialize model parameters with dummy input
variables = model.init(params_key, x_dummy)
print("Model initialized.")
print("Param structure:", jax.tree_util.tree_map(lambda x: x.shape, variables))

# define data loader
def data_loader(dataset, batch_size, shuffle = True):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        batch_image = [dataset[i][0] for i in batch_indices]
        batch_coefficients = [dataset[i][1] for i in batch_indices]
        yield jnp.array(batch_image), jnp.array(batch_coefficients)

# define loss function (simple l2 difference of coefficients)
def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
    preds = model.apply({'params': params}, inputs)  # preds: (batch, 6, 2)
    real_diffs = preds[..., 0] - true_coeffs[..., 0]
    imag_diffs = preds[..., 1] - true_coeffs[..., 1]
    sq_diffs = real_diffs**2 + imag_diffs**2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))  # sum over coefficients
    return direct_loss, direct_loss

gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 0.01)

# create optax optimizer: gradient clip -> adamw
fixed_learning_rate = config['learning_rate']['fixed']
opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate = fixed_learning_rate, weight_decay = l2_reg_weight)
)

state = train_state.TrainState.create(apply_fn = model.apply, params = variables['params'], tx = opt)

loss_history = []
test_loss_history = []

with open("training_losses_unet.csv", "w", newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])

for epoch in tqdm(range(config["optimizer"]["num_epochs"]), desc = "Training", position = 0):
    batch_loss = 0.0
    num_batches = len(dataset) // batch_size

    for batch_image, batch_coefficients in data_loader(dataset, batch_size):
        rng_key, subkey = jax.random.split(rng_key)
        loss, grads = jax.value_and_grad(loss_fn, has_aux= True)(state.params, model, batch_image, batch_coefficients,
                                                                 deterministic = False, rng_key = subkey)
        state = state.apply_gradients(grads = grads)
        batch_loss += loss[0]

    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    if test_dataset:
        test_loss = 0.0
        test_batches = len(test_dataset) // batch_size
        for test_image, test_coefficients in data_loader(test_dataset, batch_size, shuffle = False):
            total_test_loss, _ = loss_fn(state.params, model, test_image, test_coefficients, deterministic = True, rng_key = rng_key)
            test_loss += total_test_loss
        avg_test_loss = test_loss / test_batches
        test_loss_history.append(avg_test_loss.item())
    else:
        avg_test_loss = None

    with open("training_losses_unet.csv", "a", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_epoch_loss, avg_test_loss])

final_weights_name = f"unet_weights_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
