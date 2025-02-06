import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.nn as jnn
import numpy as np
import pandas as pd
import yaml
import optuna
import json  # Import json to save results
from datetime import datetime
from flax.training import train_state
import optax
from model import ConfigurableModel
from tqdm import tqdm
import pickle
import csv

import warnings
warnings.filterwarnings("ignore")

# Load configurations
with open("config_optuna.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set GPU device for training
gpu_id = config["gpu_id"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(f"Training on GPU {gpu_id}")

# Load data
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace("i", "j"))

label_file_path = config["paths"]["label_data_file_path"]
data_file_path = config["paths"]["signal_data_file_path"]

label_df = pd.read_csv(label_file_path, dtype=str)
data_df = pd.read_csv(data_file_path, dtype=str)
label_matrix = label_df.map(convert_to_complex).to_numpy().T
data_matrix = data_df.map(convert_to_complex).to_numpy().T

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

label_matrix_split = split_complex_to_imaginary(label_matrix)
data_matrix_split = split_complex_to_imaginary(data_matrix)

dataset = list(zip(data_matrix_split, label_matrix_split))

# Load test dataset if available
test_dataset = None
if "test_data_file_path" in config["paths"] and "test_label_file_path" in config["paths"]:
    test_label_df = pd.read_csv(config["paths"]["test_label_file_path"], dtype=str)
    test_data_df = pd.read_csv(config["paths"]["test_data_file_path"], dtype=str)
    test_label_matrix = test_label_df.map(convert_to_complex).to_numpy().T
    test_data_matrix = test_data_df.map(convert_to_complex).to_numpy().T
    test_label_matrix_split = split_complex_to_imaginary(test_label_matrix)
    test_data_matrix_split = split_complex_to_imaginary(test_data_matrix)
    test_dataset = list(zip(test_data_matrix_split, test_label_matrix_split))

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

def objective(trial):
    # Define hyperparameter search space
    architecture = [trial.suggest_int(f"layer_{i}", 8, 64) for i in range(trial.suggest_int("num_layers", 1, 5))]
    activation_fn = trial.suggest_categorical("activation", [jnp.tanh])
    batch_size = trial.suggest_categorical("batch_size", [16,32, 64, 128, 256, 512])
    l2_reg_weight = trial.suggest_loguniform("l2_reg_weight", 1e-6, 1e-2)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.8, 1.5)
    gradient_clip_value = trial.suggest_uniform("gradient_clip_value", 0.5, 5.0)

    # Initialize JAX keys
    root_key = jax.random.PRNGKey(seed=config["seed"])
    model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)

    input_shape = (batch_size, data_matrix_split.shape[1])
    variables = model.init(root_key, jnp.ones(input_shape), deterministic=True)

    def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
        preds = model.apply({"params": params}, inputs, deterministic=deterministic, rngs={"dropout": rng_key})
        preds_real, preds_imag = preds[:, :6], preds[:, 6:]
        true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
        loss_real = jnp.mean((preds_real - true_real) ** 2)
        loss_imag = jnp.mean((preds_imag - true_imag) ** 2)
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss_real + loss_imag + l2_reg_weight * l2_loss

    # Optimizer
    opt = optax.chain(
        optax.clip_by_global_norm(gradient_clip_value),
        optax.adamw(learning_rate=optax.exponential_decay(
            learning_rate, config["optimizer"]["step"], gamma, end_value=config["learning_rate"]["final"]
        ), weight_decay=l2_reg_weight)
    )

    state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=opt)

    # Training loop
    for epoch in range(config["optimizer"]["maxiter_adam"]):
        batch_loss = 0.0
        num_batches = len(dataset) // batch_size
        for batch_signal, batch_coefficients in data_loader(dataset, batch_size):
            rng_key, subkey = jax.random.split(root_key)
            loss, grads = jax.value_and_grad(loss_fn)(
                state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey
            )
            state = state.apply_gradients(grads=grads)
            batch_loss += loss
        avg_epoch_loss = batch_loss / num_batches

        # Evaluate on test dataset
        if test_dataset:
            test_batch_loss = 0.0
            num_test_batches = len(test_dataset) // batch_size
            for test_batch_signal, test_batch_coefficients in data_loader(test_dataset, batch_size, shuffle=False):
                test_loss = loss_fn(state.params, model, test_batch_signal, test_batch_coefficients, deterministic=True, rng_key=rng_key)
                test_batch_loss += test_loss
            avg_test_loss = test_batch_loss / num_test_batches
            trial.report(avg_test_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return avg_test_loss

# Create and optimize the study
print("Starting Optuna study...")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

# Add tqdm progress bar to track trials
with tqdm(total=40, desc="Optuna Trials", position=0) as pbar:
    def callback(study, trial):
        pbar.set_postfix({"trial": trial.number, "loss": trial.value})
        pbar.update(1)

    study.optimize(objective, n_trials=40, callbacks=[callback])

# Save the best parameters and optimal test loss
best_params = study.best_params
best_params["optimal_test_loss"] = study.best_value

output_filename = "best_params.json"
with open(output_filename, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"Best parameters and test loss saved to {output_filename}")

# Print best hyperparameters
print("Best Hyperparameters:", best_params)
