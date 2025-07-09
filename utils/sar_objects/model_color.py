import flax.linen as nn
import jax.numpy as jnp
from jax.nn import leaky_relu


class ConfigurableModel(nn.Module):
    architecture: list
    activation_fn: lambda x: leaky_relu(x, 0.01)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = x.reshape((x.shape[0], -1))  # Flatten (signal_length × 2)
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(12)(x)  # Final output: 6 real + 6 imaginary
        return x

class ConfigurableModelSingle(nn.Module):
    architecture: list
    activation_fn: callable
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = x.reshape((x.shape[0], -1))
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(1)(x)  # Single real output
        return x
