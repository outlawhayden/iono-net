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
with open("config_unet.yaml", "r") as f:
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

# load model architecture, instantiate
model_config = config["model_config"]
batch_size = config["training"]["batch_size"]

model = UNet1D(
    down_channels=model_config["down_channels"],
    bottleneck_channels=model_config["bottleneck_channels"],
    up_channels=model_config["up_channels"],
    output_dim=model_config["output_dim"]
)

signal_length = data_matrix.shape[0]  # should be 1040
x_dummy = jnp.ones((batch_size, signal_length))

# Initialize model parameters with dummy input
params = model.init(params_key, x_dummy)
print("Model initialized.")
print("Param structure:", jax.tree_util.tree_map(lambda x: x.shape, params))
