import os
# jax is the nn library i'm using here (as opposed to pytorch). flax is some people who made their own improvements but they're compatible
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import vmap
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from flax.training import train_state
import optax
from model_color import ConfigurableModel # the model class. color here is for stacking the re,im parts in the 'color' channel
from tqdm import tqdm
import pickle
import json
import csv
# patrick's definitions for some functions. theoretically these aren't used anymore
from Helper import *
from Image import *
from Psi import *
from Optimize import *

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# if I want to keep training from weights that were saved somewhere else (with the same architecture), I can specify it here
cached_weights_path = "model_weights_90.pkl"

with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

# setting random numbers for reproducibility
root_key = jax.random.PRNGKey(seed=config['seed'])
main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

# getting the architecture and activation function from the configuration file
architecture = config['model']['architecture']
activation_name = config['model']['activation']
if hasattr(jnp, activation_name):
    activation_fn = getattr(jnp, activation_name)
elif hasattr(nn, activation_name):
    activation_fn = getattr(nn, activation_name)
else:
    raise ValueError(f"Activation function '{activation_name}' not found.")

# initializing a model class object
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

# === Data preprocessing ===

# python writes out complex numbers to a csv using i, but they need to be in a certain format to become 'complex' number objects
def convert_to_complex(s):
    try:
        return complex(s.replace('i', 'j'))
    except Exception:
        return np.nan

# normalizing a matrix to unit range
def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    return matrix / amp_max


# stack real and imaginary components of a matrix in a new dimension
def stack_real_imag_as_channels(matrix):
    return np.stack([matrix.real, matrix.imag], axis=-1)

# stack real and imaginary components of a matrix vertically - what we're no longer using
def split_complex_to_imaginary(matrix):
    return np.concatenate([matrix.real, matrix.imag], axis=-1)


# read in labels and data from csv files
label_df = pd.read_csv(config['paths']['label_data_file_path'], dtype=str)
data_df = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str)


# reshape them, turn them into python objects, and normalize them.
label_matrix = label_df.map(convert_to_complex).to_numpy().T
label_max = np.max(np.abs(label_matrix))
#label_matrix /= label_max
label_matrix = label_matrix # turning off normalization for now on the Ψ coefficients. maybe it helps?
label_matrix_split = split_complex_to_imaginary(label_matrix)

data_matrix = normalize_complex_to_unit_range(data_df.map(convert_to_complex).to_numpy())
data_matrix_split = stack_real_imag_as_channels(data_matrix)

data_matrix_split = jnp.array(data_matrix_split)
label_matrix_split = jnp.array(label_matrix_split)

# create a 'dataset' of input -> label together so they can be indexed at the same time
dataset = (data_matrix_split, label_matrix_split)

# === Load test set ===

# same process, but with testing data! has to remain separate throughout the entire process
test_label_df = pd.read_csv(config['paths']['test_label_file_path'], dtype=str)
test_data_df = pd.read_csv(config['paths']['test_data_file_path'], dtype=str)

test_label_matrix = test_label_df.map(convert_to_complex).to_numpy().T / label_max
test_data_matrix = normalize_complex_to_unit_range(test_data_df.map(convert_to_complex).to_numpy())

test_label_matrix_split = split_complex_to_imaginary(test_label_matrix)
test_data_matrix_split = stack_real_imag_as_channels(test_data_matrix)

test_dataset = (jnp.array(test_data_matrix_split), jnp.array(test_label_matrix_split))

# === Load resources ===
# load domain, setup information about dataset
x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
with open(config['paths']['setup_file_path'], "r") as f:
    setup = json.load(f)
F, ionoNHarm, xi = setup['F'], setup['ionoNharm'], setup['xi']
zero_pad = 50
dx = 0.25
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()

# function to focus an image given two sets of Ψ, and calculate the L4 distance between them
def calculate_l4_norm(signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    # convert signal into complex number objects
    signal_complex = signal_vals[:, 0] + 1j * signal_vals[:, 1]
    full_len = signal_complex.shape[0]

    # Create x_range dynamically - it's just a 0.25 spacing
    x_range = jnp.arange(full_len) * DX

    # this is just to double check if something got dropped so the sizes don't line up - should just return NONE now
    start = 4 * zero_pad
    end = -4 * zero_pad if -4 * zero_pad != 0 else None
    signal_trimmed = signal_complex[start:end]
    x_trimmed = x_range[start:end]

    # make sure domain sizes match. just lets me know if something is broken and the domain sizes don't line up for some reason
    signal_trimmed = jnp.asarray(signal_trimmed).ravel()
    x_trimmed = jnp.asarray(x_trimmed).ravel()
    assert signal_trimmed.shape == x_trimmed.shape, f"Shape mismatch after trim: {signal_trimmed.shape} vs {x_trimmed.shape}"

    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

    # create image (using jax numpy or jnp so it can be done on the graphics cards fast), and use trapezoid rule to approx integral
    def evaluate_single(y):
        base = y + offsets
        real_interp = jnp.interp(base, x_trimmed, jnp.real(signal_trimmed))
        imag_interp = jnp.interp(base, x_trimmed, jnp.imag(signal_trimmed))
        signal_interp = real_interp + 1j * imag_interp
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_cos = jnp.sum(preds_real[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_sin = jnp.sum(preds_imag[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        psi_vals = jnp.exp(1j * (psi_cos - psi_sin))
        integrand = waveform * signal_interp * psi_vals
        return jnp.trapezoid(jnp.real(integrand), dx=DX) + 1j * jnp.trapezoid(jnp.imag(integrand), dx=DX)

    # use vmap (virtual map), allows the function to just be compiled once instead of every time we load it into memory
    image_vals = vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4)



# === Loss function ===
# defining loss (objective) function for the model
def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight,
            F, dx, xi, zero_pad):
    # get predicted value, split up into real and imaginary parts. calculate l2 loss of coefficients
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    sq_diffs = (preds_real - true_real) ** 2 + (preds_imag - true_imag) ** 2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    # first and second derivatives of Ψ loss term, which we don't use now but I kept it in (weight is set to 0 in config file)
    d1_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 2) * sq_diffs, axis=1))
    d2_loss = jnp.mean(jnp.sum((jnp.arange(6) ** 4) * sq_diffs, axis=1))
    # add to loss term a component for how big the actual weights of the model are, so it doesn't get too complicated. 
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    # adding a bit of amplitude difference between predicted and real Ψ to the loss function - trying to nudge it away from 0. seems to not have worked
    pred_amp = jnp.sqrt(preds_real ** 2 + preds_imag ** 2)
    true_amp = jnp.sqrt(true_real ** 2 + true_imag ** 2)
    amplitude_loss = jnp.mean((pred_amp - true_amp) ** 2)

    # add l4 loss for each element (of the entire batch). if the add_l4 flag is false, just returns 0 (is a lot faster if we skip this step, but we don't get that term)
    if add_l4:
        batch_size = inputs.shape[0]
        #sample_size = min(16, batch_size)
        #sample_indices = jax.random.choice(rng_key, batch_size, shape=(sample_size,), replace=False)
        sample_indices = jnp.arange(inputs.shape[0])
        def compute_single_l4(index):
            signal_sample = inputs[index]
            preds_real_sample = preds_real[index]
            preds_imag_sample = preds_imag[index]
            return calculate_l4_norm(signal_sample, preds_real_sample, preds_imag_sample,
                                     kpsi_values, ionoNHarm, F, dx, xi, zero_pad)

        loss_l4 = jnp.mean(jax.vmap(compute_single_l4)(sample_indices))
    else:
        loss_l4 = 0.0
    # return weighted sum of all of these components. 'fourier' weight is the coefficients of Ψ, d1 and d2 are the first and second derivatives of Ψ, l2_reg is the weights of the model,
    # l_4 loss is the L4 between the different focused images, amplitude_loss is that small term I added to weight by the amplitudes of Ψ
    return (fourier_weight * direct_loss +
            fourier_d1_weight * d1_loss +
            fourier_d2_weight * d2_loss +
            l2_reg_weight * l2_loss +
            l4_weight * loss_l4 +
            0.2 * amplitude_loss)



# === JIT-compiled training step ===
# each step we iterate on the model is written this way so it can be JIT (just-in-time compiled) - since all of the arrays in this are dynamic. 
# note that in writing it in this way doesn't require
# any for-loops in the code which is what we want to avoid
@jax.jit
def train_step(state, batch_signal, batch_coeffs, rng_key):
    def loss_wrapper(params):
        return loss_fn(params, model, batch_signal, batch_coeffs, False, rng_key,
                       ionoNHarm, kpsi_values, True,
                       l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight,
                       F, dx, xi, zero_pad)
    loss, grads = jax.value_and_grad(loss_wrapper)(state.params)
    return state.apply_gradients(grads=grads), loss


# === Training setup ===
# get all of the weights for the linear combination of the loss terms from the configuration file
gradient_clip_value = config['training'].get('gradient_clip_value', 1.0)
l2_reg_weight = config['training'].get('l2_reg_weight', 1e-4)
l4_weight = config['training'].get('l4_reg_weight', 1e-3)
fourier_weight = config['training'].get('fourier_weight', 1e-3)
fourier_d1_weight = config['training'].get('fourier_d1_weight', 1e-3)
fourier_d2_weight = config['training'].get('fourier_d2_weight', 1e-3)

# this is a learning rate scheduler - don't worry too much about this, just that the learning rate decreases according to some function across time
initial_lr = config['learning_rate']['fixed']
warmup_steps = config['learning_rate'].get('warmup_steps', 10)
total_steps = config['optimizer']['maxiter_adam']

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=initial_lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=initial_lr * 0.01
)

# this is just saying since check if any of the gradients are above gradient_clip_value, clip them there, and then do the back-propagation. 
# keeps the gradients from exploding towards infinity if something goes wrong
opt = optax.chain(
    optax.clip_by_global_norm(gradient_clip_value),
    optax.adamw(learning_rate=schedule, weight_decay=l2_reg_weight)
)

# initializing the models - set the initial weights to all 1
input_shape = (config['training']['batch_size'], data_matrix_split.shape[1], data_matrix_split.shape[2])
dummy_input = jnp.ones(input_shape)
variables = model.init(main_key, dummy_input, deterministic=True)

# if we have weights we want to use to initialize instead, we can use those here
if os.path.exists(cached_weights_path):
    with open(cached_weights_path, "rb") as f:
        loaded_params = pickle.load(f)
    state = train_state.TrainState.create(apply_fn=model.apply, params=loaded_params, tx=opt)
else:
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

# === Data loader ===
# this is a driver that gets a batch of a certain size from the dataset we created (if shuffled, its a random subset)
def data_loader(inputs, labels, batch_size, shuffle=True):
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(inputs)- batch_size + 1, batch_size):
        batch_idx = indices[start:start+batch_size]
        yield inputs[batch_idx], labels[batch_idx]

# === Training loop ===
loss_history = []
test_loss_history = []

# open a file to save the losses in as a csv
with open("training_losses_color.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])


for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training"): # just defines a progess bar to print so i can keep an eye on it
    batch_loss = 0.0
    num_batches = len(dataset[0]) // config['training']['batch_size']
    # iterate over the batch, take a train step (which returns the loss), and add it to the loss for the batch
    for batch_signal, batch_coeffs in data_loader(*dataset, config['training']['batch_size']):
        rng_key, subkey = jax.random.split(rng_key)
        state, loss = train_step(state, batch_signal, batch_coeffs, subkey)
        batch_loss += loss
    # calculate the average loss for the batch, and then save it out
    avg_epoch_loss = batch_loss / num_batches
    loss_history.append(avg_epoch_loss.item())

    # do the same thing for the 'testing' data - ie note we just compute the loss function directly, and don't use these in computing the gradients
    test_loss = 0.0
    test_batches = len(test_dataset[0]) // config['training']['batch_size']
    for test_signal, test_coeffs in data_loader(*test_dataset, config['training']['batch_size'], shuffle=False):
        test_batch_loss = loss_fn(
            state.params, model, test_signal, test_coeffs,
            True, rng_key, ionoNHarm, kpsi_values, True,
            l2_reg_weight, fourier_weight, fourier_d1_weight, fourier_d2_weight, l4_weight,
            F, dx, xi, zero_pad)
        test_loss += test_batch_loss

    avg_test_loss = test_loss / test_batches
    test_loss_history.append(avg_test_loss.item())

    # save out the information by writing to the csv - do this after everything is computed since this can't happen on the GPUs
    with open("training_losses_color_smaller.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_epoch_loss, avg_test_loss])

    # print the information out
    print(f"Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.6f}, Test Loss = {avg_test_loss:.6f}")


# once we're done, save out the weights for the model as a .pkl (pickle) file which is just a compressed data format. JAX and other libraries know how to read this - if we
# wanted to keep training on these weights later, as long as the architecture is the same we can use this file. this is also what we use for inference, we load these weights
# and then just data, and don't do any training to see how good the focusing is
final_weights_name = f"model_weights_color_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(final_weights_name, "wb") as f:
    pickle.dump(state.params, f)
print(f"Training complete. Model weights saved as '{final_weights_name}'.")
