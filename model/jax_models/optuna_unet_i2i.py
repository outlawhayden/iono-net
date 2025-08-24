# optuna_train_unet_i2i.py

import os
import optuna
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax.training import train_state
from sklearn.model_selection import KFold
import optax
from datetime import datetime
from tqdm import trange
import csv

from UNet1D_i2i import UNet1D_i2i
from Helper import *
from Image import *
from Psi import *
from Optimize import *

jax.config.update("jax_enable_x64", True)
os.environ["JAX_TRACEBACK_FILTERING"] = "on"

# === Setup ===
with open("config_unet_i2i.yaml", "r") as f:
    base_config = yaml.safe_load(f)

log_file = "unet_i2i_optuna_trials.csv"
with open(log_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "fold", "epoch", "train_loss", "test_loss"])


# === Utility Functions ===
def convert_to_complex(s):
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace('i', 'j'))

def stack_real_imag_as_channels(complex_array):
    return np.concatenate([complex_array.real[..., np.newaxis],
                           complex_array.imag[..., np.newaxis]], axis=-1)

def pad_signal(x, divisor):
    length = x.shape[1]
    remainder = length % divisor
    if remainder == 0:
        return x
    pad_total = divisor - remainder
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))

def data_loader(dataset, batch_size, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(dataset), batch_size):
        excerpt = indices[start:start + batch_size]
        yield jnp.array([dataset[i][0] for i in excerpt]), jnp.array([dataset[i][1] for i in excerpt])

# === Loss ===
def loss_fn(params, model, inputs, true_images, deterministic, rng_key, amp_weight, l4_weight):
    preds = model.apply({'params': params}, inputs)
    real_diffs = preds[..., 0] - true_images[..., 0]
    imag_diffs = preds[..., 1] - true_images[..., 1]
    sq_diffs = real_diffs**2 + imag_diffs**2
    amp_true = jnp.sqrt(true_images[..., 0]**2 + true_images[..., 1]**2)
    weight = amp_true / (jnp.max(amp_true, axis=1, keepdims=True) + 1e-8)
    weight = jnp.clip(weight, 1e-3, 1.0)
    weighted_sq_diffs = weight * sq_diffs
    weighted_loss = jnp.mean(jnp.sum(weighted_sq_diffs, axis=1))
    amp_pred = jnp.sqrt(preds[..., 0]**2 + preds[..., 1]**2)
    amp_loss = jnp.mean(jnp.sum((amp_pred - amp_true)**2, axis=1))
    l4_loss = jnp.mean(jnp.sum(jnp.abs(real_diffs + 1j * imag_diffs) ** 4, axis=1))
    total_loss = weighted_loss + amp_weight * amp_loss + l4_weight * l4_loss
    return total_loss, weighted_loss

# === Objective ===
def objective(trial):
    config = base_config.copy()

    # Hyperparameter suggestions
    learning_rate = trial.suggest_loguniform("lr", 1e-7, 1e-5)
    l4_weight = trial.suggest_uniform("l4_weight", 0.0, 0.01)
    amp_weight = trial.suggest_uniform("amp_weight", 0.0, 0.1)
    l2_weight = trial.suggest_loguniform("l2_weight", 1e-9, 1e-4)

    print(f"\n[Trial {trial.number}] Starting with:")
    print(f"  lr={learning_rate:.2e}, l4_weight={l4_weight:.3f}, amp_weight={amp_weight:.3f}, l2_weight={l2_weight:.2e}")

    # Load and preprocess full dataset
    label_raw = pd.read_csv(config['paths']['label_data_file_path']).map(convert_to_complex).to_numpy().T
    data_raw = pd.read_csv(config['paths']['data_file_path']).map(convert_to_complex).to_numpy().T
    data_mean, data_std = np.mean(data_raw), np.std(data_raw)
    label_mean, label_std = np.mean(label_raw), np.std(label_raw)
    data = stack_real_imag_as_channels(((data_raw - data_mean) / data_std).T)
    labels = stack_real_imag_as_channels(((label_raw - label_mean) / label_std).T)

    batch_size = config['training']['batch_size']
    depth = len(config['model_config']['down_channels'])
    divisor = 2 ** depth
    data = pad_signal(data, divisor)
    labels = pad_signal(labels, divisor)
    full_dataset = list(zip(data, labels))

    kf = KFold(n_splits=4, shuffle=True, random_state=config['seed'])
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        train_data = [full_dataset[i] for i in train_idx]
        val_data = [full_dataset[i] for i in val_idx]

        model = UNet1D_i2i(
            down_channels=config["model_config"]["down_channels"],
            bottleneck_channels=config["model_config"]["bottleneck_channels"],
            up_channels=config["model_config"]["up_channels"],
            output_dim=2
        )

        rng = jax.random.PRNGKey(config['seed'] + fold_idx)
        init_key, train_key = jax.random.split(rng)
        dummy_input = jnp.ones((batch_size, data.shape[1], 2))
        params = model.init(init_key, dummy_input)["params"]

        tx = optax.chain(
            optax.clip_by_global_norm(config['training']['gradient_clip_value']),
            optax.adamw(learning_rate=learning_rate, weight_decay=l2_weight)
        )

        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        num_epochs = 1000
        patience = 20
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        pbar = trange(num_epochs, desc=f"Trial {trial.number} Fold {fold_idx}", position=0, leave=False)
        for epoch in pbar:
            train_losses = []
            for batch_x, batch_y in data_loader(train_data, batch_size):
                train_key, subkey = jax.random.split(train_key)
                (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params, model, batch_x, batch_y, False, subkey, amp_weight, l4_weight
                )
                train_losses.append(loss.item())
                state = state.apply_gradients(grads=grads)

            avg_train_loss = np.mean(train_losses)

            val_losses = []
            for val_x, val_y in data_loader(val_data, batch_size, shuffle=False):
                val_loss, _ = loss_fn(state.params, model, val_x, val_y, True, train_key, amp_weight, l4_weight)
                val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)

            # Write to CSV
            with open(log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial.number, fold_idx, epoch, avg_train_loss, avg_val_loss])

            # Early stopping check
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"[Trial {trial.number} Fold {fold_idx}] Early stopping at epoch {epoch}")
                break


        fold_losses.append(avg_val_loss)

    final_loss = np.mean(fold_losses)
    print(f"[Trial {trial.number}] Mean CV Loss: {final_loss:.6f}")
    return final_loss

# === Run Study ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)

# === Save results ===
with open("best_unet_i2i_trial.yaml", "w") as f:
    yaml.dump(study.best_params, f)
print("Best hyperparameters:", study.best_params)
print("Best test loss:", study.best_value)
