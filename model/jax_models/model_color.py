import flax.linen as nn
import jax.numpy as jnp
from jax.nn import leaky_relu
from jax.nn import tanh

class ConfigurableModel(nn.Module):
    architecture: list # can call architecture from list in config file. list of integers for layers, and size of each layer
    # for now i had to set leaky_relu directly since it has a parameter in it (negative slope). otherwise just set it to callable
    activation_fn: callable = lambda x: leaky_relu(x, 0.001)
    dropout_rate: float = 0.2

    # quick compilation using nn.compact
    @nn.compact
    # when you call configurable model
    def __call__(self, x, deterministic: bool):
        x = x.reshape((x.shape[0], -1))  # Flatten (signal_length × 2)
        for layer_size in self.architecture: # iterate over layers
            # dense (fully connected layer)
            x = nn.Dense(layer_size)(x)
            # normalize the weights on each layer
            x = nn.LayerNorm()(x)
            # run through activation function σ 
            x = self.activation_fn(x)
            # do droput (randomly pause different weights) if needed
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(12)(x)  # Final output: 6 real + 6 imaginary
        return x

# this is the same model, but just one real output which i was using to test if Ψ is just one coefficient and the other 11 are zero
class ConfigurableModelSingle(nn.Module):
    architecture: list
    activation_fn: callable = lambda x: leaky_relu(x, 0)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = x.reshape((x.shape[0], -1))
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = nn.LayerNorm()(x)  # Layer normalization
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(1)(x)  # Single real output
        return x
