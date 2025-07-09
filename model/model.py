import flax.linen as nn
import jax.numpy as jnp
from jax.nn import initializers

class ConfigurableModel(nn.Module):
    architecture: list
    activation_fn: callable
    dropout_rate: float = 0.0
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for layer_size in self.architecture:
            x = nn.Dense(layer_size, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(12, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x

class ConfigurableModelSingle(nn.Module):
    architecture: list
    activation_fn: callable
    dropout_rate: float = 0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(1)(x)  # Final layer with 1 output (no imaginary component)
        return x
    

