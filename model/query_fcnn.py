import pickleimport jax
import jax.numpy as jnp

pkl_path = 'model_data.pkl'

with open(pkl_path', 'rb') as f:
    model_params = pickle.load(f)


