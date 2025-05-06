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

# load in data according to file paths
label_matrix = pd.read_csv(label_file_path).map(convert_to_complex).to_numpy().T
data_matrix = pd.read_csv(data_file_path).map(convert_to_complex).to_numpy().T
data_matrix = np.abs(data_matrix[:-1,:])
x_range = pd.read_csv(x_range_file_path).iloc[:,0].values #1040 long


# convert labels from 6 complex -> 12 real

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

label_matrix = split_complex_to_imaginary(label_matrix)

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

signal_length = data_matrix.shape[0]+1  # should be 1040
x_dummy = jnp.ones((batch_size, signal_length))

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
    # preds = model.apply({'params':params}, inputs, deterministic = deterministic)
    preds = model.apply({'params':params}, inputs)
    preds_real, preds_imag = preds[:,:6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs [:,6:]
    real_diffs = preds_real - true_real
    imag_diffs = preds_imag - true_imag
    sq_diffs = real_diffs ** 2 + imag_diffs ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis = 1))
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

with open("training_losses_unet.csv", "w", newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss"])

batch_size = config["training"]["batch_size"]

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

    with open("training_losses_unet.csv", "a", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_epoch_loss])


