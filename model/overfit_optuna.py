import os
import optuna
import pickle
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from datetime import datetime
import json
import csv

from model_color import ConfigurableModel
from Helper import *
from Image import *
from Psi import *
from Optimize import *

with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_data(config):
    def convert(s): return complex(s.replace('i', 'j'))
    def norm_unit(x): return x / np.max(np.abs(x), axis=1, keepdims=True)
    def to_channels(x): return np.stack([x.real, x.imag], axis=-1)
    def to_flat(x): return np.concatenate([x.real, x.imag], axis=-1)

    NUM = 1024
    label = pd.read_csv(config['paths']['label_data_file_path'], dtype=str).iloc[:, :NUM]
    signal = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str).iloc[:NUM, :]
    label = label.map(convert).to_numpy().T
    signal = norm_unit(pd.DataFrame(signal.map(convert)).to_numpy())

    test_label = pd.read_csv(config['paths']['test_label_file_path'], dtype=str).iloc[:, :NUM]
    test_signal = pd.read_csv(config['paths']['test_data_file_path'], dtype=str).iloc[:NUM, :]
    test_label = test_label.map(convert).to_numpy().T
    test_signal = norm_unit(test_signal.map(convert).to_numpy())

    return (jnp.array(to_channels(signal)), jnp.array(to_flat(label)),
            jnp.array(to_channels(test_signal)), jnp.array(to_flat(test_label)))


def calculate_l4_norm(signal_vals, preds_real, preds_imag, kpsi_values, ionoNHarm, F, DX, xi, zero_pad):
    signal_complex = signal_vals[:, 0] + 1j * signal_vals[:, 1]
    full_len = signal_complex.shape[0]
    x_range = jnp.arange(full_len) * DX

    start = 4 * zero_pad
    end = -4 * zero_pad if -4 * zero_pad != 0 else None
    signal_trimmed = signal_complex[start:end]
    x_trimmed = x_range[start:end]

    signal_trimmed = jnp.asarray(signal_trimmed).ravel()
    x_trimmed = jnp.asarray(x_trimmed).ravel()
    assert signal_trimmed.shape == x_trimmed.shape, f"Shape mismatch after trim: {signal_trimmed.shape} vs {x_trimmed.shape}"

    window_size = int(F / DX) + 1
    offsets = jnp.linspace(-F / 2, F / 2, window_size)

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

    image_vals = jax.vmap(evaluate_single)(x_trimmed)
    return jnp.sum(jnp.abs(image_vals) ** 4)


def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key,
            ionoNHarm, kpsi_values, add_l4, l2_reg_weight, fourier_weight, l4_weight,
            F, dx, xi, zero_pad, amplitude_weight):
    preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
    preds_real, preds_imag = preds[:, :6], preds[:, 6:]
    true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
    sq_diffs = (preds_real - true_real)**2 + (preds_imag - true_imag)**2
    direct_loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    pred_amp = jnp.sqrt(preds_real ** 2 + preds_imag ** 2)
    true_amp = jnp.sqrt(true_real ** 2 + true_imag ** 2)
    amplitude_loss = jnp.mean((pred_amp - true_amp) ** 2)

    if add_l4:
        def l4_one(i):
            sig = inputs[i]
            pr, pi = preds_real[i], preds_imag[i]
            return calculate_l4_norm(sig, pr, pi, kpsi_values, ionoNHarm, F, dx, xi, zero_pad)
        loss_l4 = jnp.mean(jax.vmap(l4_one)(jnp.arange(inputs.shape[0])))
    else:
        loss_l4 = 0.0

    total = fourier_weight * direct_loss + l2_reg_weight * l2_loss + l4_weight * loss_l4 + amplitude_loss * amplitude_weight
    return total, direct_loss

def train_step(state, batch_signal, batch_coeffs, rng_key, model, **loss_kwargs):
    def wrapper(params):
        return loss_fn(params, model, batch_signal, batch_coeffs, False, rng_key, **loss_kwargs)
    (loss, direct), grads = jax.value_and_grad(wrapper, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, direct

def data_loader(X, Y, batch_size, shuffle=True, noise_std=0.01):
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    for start in range(0, len(X) - batch_size + 1, batch_size):
        i = idx[start:start + batch_size]
        xb, yb = X[i], Y[i]
        if shuffle:
            noise = np.random.normal(0, noise_std, xb.shape).astype(np.float32)
            xb = xb + noise
        yield xb, yb


def objective(trial):
    layers = trial.suggest_int("n_layers", 3, 12)
    width = trial.suggest_categorical("layer_width", [64, 128, 256, 512, 1024])
    architecture = [width] * layers

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    l2_reg_weight = trial.suggest_float("l2_reg_weight", 1e-5, 1e-1, log=True)
    l4_weight = trial.suggest_float("l4_weight", 0.0, 1.0)
    fourier_weight = trial.suggest_float("fourier_weight", 1e-4, 1.0, log=True)
    amplitude_weight = trial.suggest_float("amplitude_weight", 1e-4, 1.0, log = True)


    batch_size = config['training']['batch_size']
    max_epochs = 400

    train_X, train_Y, test_X, test_Y = load_data(config)

    with open(config['paths']['setup_file_path']) as f: setup = json.load(f)
    F, xi, ionoNHarm = setup["F"], setup["xi"], setup["ionoNharm"]
    kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()
    dx, zero_pad = 0.25, 50

    model = ConfigurableModel(architecture=architecture, activation_fn=jnp.tanh)
    dummy_input = train_X[:batch_size].reshape(batch_size, -1)
    rng = jax.random.PRNGKey(config['seed'])
    init_vars = model.init(rng, dummy_input, deterministic=True)

    schedule = optax.warmup_cosine_decay_schedule(0.0, learning_rate, 10, max_epochs, learning_rate * 0.01)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=l2_reg_weight))
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_vars['params'], tx=tx)

    for epoch in range(max_epochs):
        for xb, yb in data_loader(train_X, train_Y, batch_size):
            rng, subkey = jax.random.split(rng)
            state, _, _ = train_step(state, xb, yb, subkey, model,
                                     ionoNHarm=ionoNHarm, kpsi_values=kpsi_values,
                                     add_l4=False, l2_reg_weight=l2_reg_weight,
                                     fourier_weight=fourier_weight, l4_weight=l4_weight,
                                     F=F, dx=dx, xi=xi, zero_pad=zero_pad, amplitude_weight = amplitude_weight)

    total_loss = 0.0
    count = 0
    for xb, yb in data_loader(test_X, test_Y, batch_size, shuffle=False):
        _, direct_loss = loss_fn(state.params, model, xb, yb, True, rng,
                                ionoNHarm, kpsi_values, True,
                                l2_reg_weight, fourier_weight, l4_weight,
                                F, dx, xi, zero_pad, amplitude_weight)
        total_loss += direct_loss.item()
        count += 1

    with open("optuna_trials_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "trial", "test_loss", "n_layers", "layer_width", "learning_rate",
                "l2_reg_weight", "l4_weight", "fourier_weight", "amplitude_weight"
            ])
        writer.writerow([
            trial.number, total_loss / count, layers, width, learning_rate,
            l2_reg_weight, l4_weight, fourier_weight, amplitude_weight
        ])

    return total_loss / count

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open("optuna_best_params_color.json", "w") as f:
        json.dump(trial.params, f, indent=2)
