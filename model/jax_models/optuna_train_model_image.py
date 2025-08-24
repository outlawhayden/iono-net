import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
import optuna
import yaml
import pickle
import optax
import json
import csv
from flax.training import train_state
from datetime import datetime
from model import ConfigurableModel
from tqdm import tqdm
from Helper import *
from Image import *
from Psi import *
from Optimize import *

# === Setup
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# === Config
with open("config_image.yaml", "r") as f:
    base_config = yaml.safe_load(f)

label_matrix = pd.read_csv(base_config['paths']['label_data_file_path'], dtype=str).map(
    lambda s: complex(s.replace('i', 'j')) if s != "NaNNaNi" else 0).to_numpy().T
data_matrix = pd.read_csv(base_config['paths']['signal_data_file_path'], dtype=str).map(
    lambda s: complex(s.replace('i', 'j')) if s != "NaNNaNi" else 0).to_numpy().T
x_range = pd.read_csv(base_config['paths']['x_range_file_path']).iloc[:, 0].values

split_complex = lambda c: np.concatenate([c.real, c.imag], axis=-1)
dataset = list(zip(split_complex(data_matrix), split_complex(label_matrix)))

test_dataset = None
if "test_data_file_path" in base_config['paths']:
    test_data = pd.read_csv(base_config['paths']['test_data_file_path'], dtype=str).map(
        lambda s: complex(s.replace('i', 'j')) if s != "NaNNaNi" else 0).to_numpy().T
    test_label = pd.read_csv(base_config['paths']['test_label_file_path'], dtype=str).map(
        lambda s: complex(s.replace('i', 'j')) if s != "NaNNaNi" else 0).to_numpy().T
    test_dataset = list(zip(split_complex(test_data), split_complex(test_label)))

setup = json.load(open(base_config['paths']['setup_file_path']))
kpsi_values = pd.read_csv(base_config['paths']['kpsi_file_path']).values
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
dx = 0.25
zero_pad = 50
seed = base_config['seed']
batch_size = base_config['training']['batch_size']

# === Fast image loss
def image_loss(x_range, signal, pr, pi, tr, ti):
    signal_c = signal[:1441] + 1j * signal[1441:]
    signal_trimmed = signal_c[4 * zero_pad : -4 * zero_pad]
    x_trimmed = x_range[4 * zero_pad : -4 * zero_pad]
    offsets = jnp.linspace(-F/2, F/2, int(F/dx)+1)

    def eval_point(y, pr, pi, tr, ti):
        base = y + offsets
        interp = jnp.interp(base, x_trimmed, signal_trimmed)
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_pred = jnp.exp(1j * (jnp.sum(pr[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0) +
                                  jnp.sum(pi[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)))
        psi_true = jnp.exp(1j * (jnp.sum(tr[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0) +
                                  jnp.sum(ti[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)))
        img_pred = jnp.trapezoid(waveform * interp * psi_pred, dx=dx) / F
        img_true = jnp.trapezoid(waveform * interp * psi_true, dx=dx) / F
        return jnp.abs(img_pred - img_true) ** 2

    return jnp.trapezoid(jax.vmap(lambda y: eval_point(y, pr, pi, tr, ti))(x_trimmed), x_trimmed)

# === Loss
def loss_fn(params, model, x, y, deterministic, rng, fw, iw):
    pred = model.apply({'params': params}, x, deterministic=deterministic, rngs={'dropout': rng})
    pr, pi = pred[:, :6], pred[:, 6:]
    tr, ti = y[:, :6], y[:, 6:]
    mse = jnp.mean((pr - tr)**2 + (pi - ti)**2)

    def sample_image_loss(i):
        return image_loss(x_range, x[i], pr[i], pi[i], tr[i], ti[i])

    image_loss_sampled = jnp.mean(jax.vmap(sample_image_loss)(jnp.arange(min(2, x.shape[0]))))
    return fw * mse + iw * image_loss_sampled

# === DataLoader
def data_loader(dataset, batch_size):
    idxs = np.random.permutation(len(dataset))
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in idxs[i:i+batch_size]]
        yield jnp.array([b[0] for b in batch]), jnp.array([b[1] for b in batch])

# === Training Step
def train_step(state, model, x, y, rng, fw, iw):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, model, x, y, False, rng, fw, iw)
    state = state.apply_gradients(grads=grads)
    return state, loss

# === Optuna Objective
def objective(trial):
    arch = [trial.suggest_categorical("width", [64, 128, 256])] * trial.suggest_int("depth", 1, 4)
    act = trial.suggest_categorical("activation", ["relu", "tanh", "gelu"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    fw = trial.suggest_float("fw", 0.0, 1.0)
    iw = 1.0 - fw  # <-- force iw to sum to 1 - fw
    activation_fn = {
        "relu": nn.relu,
        "tanh": jnp.tanh,
        "gelu": nn.gelu,
    }[act]
    model = ConfigurableModel(architecture=arch, activation_fn=activation_fn)
    rng_key = jax.random.PRNGKey(seed)
    input_shape = (batch_size, dataset[0][0].shape[0])
    params = model.init(rng_key, jnp.ones(input_shape), deterministic=True)['params']
    tx = optax.adamw(learning_rate=lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(10):
        for xb, yb in data_loader(dataset, batch_size):
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = train_step(state, model, xb, yb, subkey, fw, iw)

        if test_dataset:
            val_loss = 0
            for xt, yt in data_loader(test_dataset, batch_size):
                val_loss += loss_fn(state.params, model, xt, yt, True, rng_key, fw, iw)
            val_loss /= max(1, len(test_dataset) // batch_size)
        else:
            val_loss = 9999.0

        trial.report(val_loss.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="psi_model_image_opt",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage="sqlite:///optuna.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30)
    with open("best_params_image.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
