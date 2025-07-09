import flax.linen as nn
import jax.numpy as jnp

class ConfigurableModel(nn.Module):
    architecture: list
    activation_fn: callable
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, deterministic: bool):
        if x.ndim == 3:  # e.g., (batch, signal_length, 2)
            x = x.reshape((x.shape[0], -1))  # → (batch, signal_length * 2)
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(12)(x)  # 6 real + 6 imag
        return x
