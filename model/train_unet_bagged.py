import os
import jax
import jax.numpy as jnp
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
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from Helper import *
from Image import *
from Psi import *
from Optimize import *

jax.config.update("jax_enable_x64", True)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

def main():
    with open("/home/houtlaw/iono-net/model/config_unet.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    def convert_to_complex(s): 
        if s == "NaNNaNi":
            return np.nan
        else:
            return complex(s.replace('i', 'j'))

    def stack_real_imag_as_channels(complex_array):
        real_part = complex_array.real[..., np.newaxis]
        imag_part = complex_array.imag[..., np.newaxis]
        return np.concatenate([real_part, imag_part], axis=-1)

    label_file_path = config['paths']['label_data_file_path']
    data_file_path = config['paths']['data_file_path']
    x_range_file_path = config['paths']['x_range_file_path']
    setup_path = config['paths']['setup_file_path']
    kpsi_values_path = config['paths']['kpsi_file_path']

    with open(setup_path) as f:
        setup = json.load(f)

    kpsi_values = pd.read_csv(kpsi_values_path).values

    label_matrix_raw = pd.read_csv(label_file_path).map(convert_to_complex).to_numpy().T
    data_matrix_raw = pd.read_csv(data_file_path).map(convert_to_complex).to_numpy().T

    data_mean = np.mean(data_matrix_raw)
    data_std = np.std(data_matrix_raw)
    data_matrix_norm = (data_matrix_raw - data_mean) / data_std
    data_matrix = stack_real_imag_as_channels(data_matrix_norm.T)

    label_mean = np.mean(label_matrix_raw)
    label_std = np.std(label_matrix_raw)
    label_matrix_norm = (label_matrix_raw - label_mean) / label_std
    label_matrix = stack_real_imag_as_channels(label_matrix_norm)

    dataset = list(zip(data_matrix, label_matrix))

    def create_bootstrap_dataset(dataset, seed, size=None):
        rng = np.random.default_rng(seed)
        n = len(dataset)
        size = size or n
        indices = rng.choice(n, size=size, replace=True)
        return [dataset[i] for i in indices]

    def train_single_model(model_idx, seed, dataset, config, data_mean, data_std, label_mean, label_std):
        print(f"Training bagged model {model_idx}")
        rng_key = jax.random.PRNGKey(seed + model_idx)
        main_key, params_key = jax.random.split(rng_key)

        boot_dataset = create_bootstrap_dataset(dataset, seed + model_idx)
        batch_size = config["training"]["batch_size"]
        model_config = config["model_config"]

        model = UNet1D(
            down_channels=model_config["down_channels"],
            bottleneck_channels=model_config["bottleneck_channels"],
            up_channels=model_config["up_channels"],
            output_dim=model_config["output_dim"]
        )

        dummy_input = jnp.ones((batch_size, boot_dataset[0][0].shape[0], 2))
        variables = model.init(params_key, dummy_input)

        opt = optax.chain(
            optax.clip_by_global_norm(config['training'].get('gradient_clip_value', 1.0)),
            optax.adamw(learning_rate=config['learning_rate']['fixed'],
                        weight_decay=config['training'].get('l2_reg_weight', 0.01))
        )
        state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=opt)

        def data_loader(dataset, batch_size, shuffle=True):
            indices = np.arange(len(dataset))
            if shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                x = jnp.array([dataset[i][0] for i in batch_idx])
                y = jnp.array([dataset[i][1] for i in batch_idx])
                yield x, y

        def loss_fn(params, model, inputs, targets, rng_key):
            preds = model.apply({'params': params}, inputs)
            real_diffs = preds[..., 0] - targets[..., 0]
            imag_diffs = preds[..., 1] - targets[..., 1]
            sq_diffs = real_diffs**2 + imag_diffs**2
            loss = jnp.mean(jnp.sum(sq_diffs, axis=1))
            return loss, loss

        for epoch in range(config["optimizer"]["num_epochs"]):
            for x_batch, y_batch in data_loader(boot_dataset, batch_size):
                rng_key, subkey = jax.random.split(rng_key)
                loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, model, x_batch, y_batch, subkey)
                state = state.apply_gradients(grads=grads)

        model_path = f"bagged_model_{model_idx}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(state.params, f)
        print(f"Model {model_idx} saved to {model_path}")
        return model_path

    n_models = config.get("bagging", {}).get("n_models", 5)

    tasks = [
        delayed(train_single_model)(
            model_idx=i,
            seed=seed,
            dataset=dataset,
            config=config,
            data_mean=data_mean,
            data_std=data_std,
            label_mean=label_mean,
            label_std=label_std
        )
        for i in range(n_models)
    ]

    results = compute(*tasks)
    print("All models trained and saved:")
    for path in results:
        print(" -", path)

    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_single_model(params, model_def, inputs):
        return model_def.apply({'params': params}, inputs)

    def ensemble_predict(models, model_def, inputs):
        preds = [predict_single_model(p, model_def, inputs) for p in models]
        stacked = jnp.stack(preds, axis=0)
        return jnp.mean(stacked, axis=0)

    if "test_data_file_path" in config['paths'] and "test_label_file_path" in config['paths']:
        print("Loading test dataset for evaluation...")
        test_label_matrix_raw = pd.read_csv(config["paths"]["test_label_file_path"]).map(convert_to_complex).to_numpy().T
        test_label_matrix_norm = (test_label_matrix_raw - label_mean) / label_std
        test_label_matrix = stack_real_imag_as_channels(test_label_matrix_norm)

        test_data_matrix_raw = pd.read_csv(config["paths"]["test_data_file_path"]).map(convert_to_complex).to_numpy().T
        test_data_matrix_norm = (test_data_matrix_raw - data_mean) / data_std
        test_data_matrix = stack_real_imag_as_channels(test_data_matrix_norm.T)

        test_dataset = list(zip(test_data_matrix, test_label_matrix))
    else:
        raise RuntimeError("No test dataset configured for evaluation.")

    model_config = config["model_config"]
    model = UNet1D(
        down_channels=model_config["down_channels"],
        bottleneck_channels=model_config["bottleneck_channels"],
        up_channels=model_config["up_channels"],
        output_dim=model_config["output_dim"]
    )

    def test_data_loader(dataset, batch_size):
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            x_batch = jnp.array([dataset[i][0] for i in range(start_idx, end_idx)])
            y_batch = jnp.array([dataset[i][1] for i in range(start_idx, end_idx)])
            yield x_batch, y_batch

    batch_size = config["training"]["batch_size"]
    bagged_model_paths = [f"bagged_model_{i}.pkl" for i in range(n_models)]
    bagged_models = [load_model(p) for p in bagged_model_paths]

    individual_losses = []

    for i, params in enumerate(bagged_models):
        losses = []
        for x_batch, y_batch in test_data_loader(test_dataset, batch_size):
            pred = predict_single_model(params, model, x_batch)
            real_diff = pred[..., 0] - y_batch[..., 0]
            imag_diff = pred[..., 1] - y_batch[..., 1]
            sq_diff = real_diff**2 + imag_diff**2
            batch_loss = jnp.mean(jnp.sum(sq_diff, axis=1))
            losses.append(batch_loss)
        model_loss = float(jnp.mean(jnp.array(losses)))
        print(f"Model {i} Test Loss: {model_loss:.6f}")
        individual_losses.append((i, model_loss))

    ensemble_losses = []
    for x_batch, y_batch in test_data_loader(test_dataset, batch_size):
        pred_batch = ensemble_predict(bagged_models, model, x_batch)
        real_diff = pred_batch[..., 0] - y_batch[..., 0]
        imag_diff = pred_batch[..., 1] - y_batch[..., 1]
        sq_diff = real_diff**2 + imag_diff**2
        batch_loss = jnp.mean(jnp.sum(sq_diff, axis=1))
        ensemble_losses.append(batch_loss)

    avg_ensemble_loss = float(jnp.mean(jnp.array(ensemble_losses)))
    print(f"\n Ensemble Average Test Loss: {avg_ensemble_loss:.6f}")

    with open("bagging_test_losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Index", "Test Loss"])
        for idx, loss in individual_losses:
            writer.writerow([idx, loss])
        writer.writerow(["Ensemble Average", avg_ensemble_loss])

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
