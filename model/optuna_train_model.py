import optuna
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from flax.training import train_state
import optax
from model import ConfigurableModel
from tqdm import tqdm
import pickle
import csv

# List available GPU devices
devices = jax.devices()
num_gpus = len(devices)
print(f"Detected {num_gpus} GPU(s): {[d.id for d in devices]}")

# Set up Optuna to use parallel execution with each job pinned to a specific GPU
def objective(trial):
    # Determine which GPU to use for this trial
    gpu_id = trial.number % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Running trial {trial.number} on GPU {gpu_id}")

    # Sample hyperparameters from Optuna
    learning_rate_initial = trial.suggest_float("learning_rate_initial", 0.0001, 0.01, log=True)
    learning_rate_gamma = trial.suggest_float("learning_rate_gamma", 0.85, 0.99)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    
    # Iterate over architectures with different layers and sizes
    num_layers = trial.suggest_int("num_layers", 3, 6)
    layer_sizes = [trial.suggest_int(f"layer_{i}_size", 128, 2048, log=True) for i in range(num_layers)]
    architecture = layer_sizes + [64, 32]  # Add fixed output layers

    # Load configurations
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Update the config with sampled hyperparameters
    config['learning_rate']['initial'] = learning_rate_initial
    config['learning_rate']['gamma'] = learning_rate_gamma
    config['training']['batch_size'] = batch_size
    config['model']['trunk_architecture'] = architecture

    # Initialize JAX keys
    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key, rng_key = jax.random.split(key=root_key, num=3)

    # Instantiate the model
    activation_fn = getattr(jnp, config['model']['trunk_activation'])
    model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)
    architecture_penalty = config['model']['architecture_penalty']

    # Load data as per original script - use file paths from config file
    label_file_path = config['paths']['label_data_file_path']
    data_file_path = config['paths']['signal_data_file_path']

    def convert_to_complex(s):
        if s == "NaNNaNi":
            return 0
        else:
            return complex(s.replace('i', 'j'))

    label_df = pd.read_csv(label_file_path, dtype=str)
    data_df = pd.read_csv(data_file_path, dtype=str)
    label_matrix = label_df.map(convert_to_complex).to_numpy().T
    data_matrix = data_df.map(convert_to_complex).to_numpy().T

    def split_complex_to_imaginary(complex_array):
        return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

    label_matrix_split = split_complex_to_imaginary(label_matrix)
    data_matrix_split = split_complex_to_imaginary(data_matrix)

    dataset = list(zip(data_matrix_split, label_matrix_split))

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

    input_shape = (batch_size, data_matrix_split.shape[1])
    variables = model.init(root_key, jnp.ones(input_shape), deterministic=True)

    def loss_fn(params, model, inputs, true_coeffs, deterministic, rng_key):
        preds = model.apply({'params': params}, inputs, deterministic=deterministic, rngs={'dropout': rng_key})
        preds_real, preds_imag = preds[:, :6], preds[:, 6:]
        true_real, true_imag = true_coeffs[:, :6], true_coeffs[:, 6:]
        loss_real = jnp.mean((preds_real - true_real) ** 2)
        loss_imag = jnp.mean((preds_imag - true_imag) ** 2)
        return loss_real + loss_imag

    opt = optax.adam(optax.exponential_decay(
        config['learning_rate']['initial'],
        config['learning_rate']['step'],
        config['learning_rate']['gamma'],
        end_value=config['learning_rate']['final']
    ))

    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

    minloss = float('inf')
    loss_history = []

    # Use tqdm for the progress bar
    for epoch in tqdm(range(config['optimizer']['maxiter_adam']), desc="Training", position=0):
        batch_loss = 0.0
        num_batches = len(dataset) // batch_size
        for batch_signal, batch_coefficients in data_loader(dataset, batch_size):
            rng_key, subkey = jax.random.split(rng_key)
            loss, grads = jax.value_and_grad(loss_fn)(
                state.params, model, batch_signal, batch_coefficients, deterministic=False, rng_key=subkey
            )
            state = state.apply_gradients(grads=grads)
            batch_loss += loss
        avg_epoch_loss = batch_loss / num_batches
        loss_history.append(avg_epoch_loss.item())
        if avg_epoch_loss < minloss:
            minloss = avg_epoch_loss

    # Define the architecture penalty
    architecture_penalty = architecture_penalty * sum(layer_sizes) * 1e-5  # Adjust weight as needed

    # Compute the final loss including the architecture penalty
    final_loss = minloss + architecture_penalty

    # Save the best loss and history
    trial.set_user_attr("loss_history", loss_history)
    trial.set_user_attr("minloss", final_loss)

    return final_loss


# Run the Optuna study with parallel trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, n_jobs=num_gpus)

# Save the best trial parameters and loss history to a YAML file
best_trial = study.best_trial
params_dict = best_trial.params
params_dict["minloss"] = best_trial.user_attrs["minloss"]
params_dict["architecture"] = [
    best_trial.params.get(f"layer_{i}_size") for i in range(best_trial.params["num_layers"])
] + [64, 32]  # Add fixed output layers

datestr = datetime.now().strftime('%Y%m%d_%H%M%S')

output_filename = f'optuna_best_params_{datestr}.yml'

# Save parameters to YAML
with open(output_filename, 'w') as outfile:
    yaml.dump(params_dict, outfile, default_flow_style=False)

print(f"Best trial parameters saved to {output_filename}")

# Export loss history to CSV
loss_history_csv_filename = f'loss_history_{datestr}.csv'
with open(loss_history_csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Loss"])
    for i, loss in enumerate(best_trial.user_attrs["loss_history"]):
        csvwriter.writerow([i + 1, loss])

print(f"Loss history saved to {loss_history_csv_filename}")

# Reinitialize the best model
activation_fn = getattr(jnp, config['model']['trunk_activation'])
best_architecture = params_dict["architecture"]
model = ConfigurableModel(architecture=best_architecture, activation_fn=activation_fn)

input_shape = (params_dict["batch_size"], data_matrix_split.shape[1])
variables = model.init(jax.random.PRNGKey(seed=0), jnp.ones(input_shape), deterministic=True)

# Save model weights to PKL
weights_filename = f'model_weights_{datestr}.pkl'
with open(weights_filename, 'wb') as weights_file:
    pickle.dump(variables['params'], weights_file)

print(f"Model weights saved to {weights_filename}")

# Final best trial summary
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
