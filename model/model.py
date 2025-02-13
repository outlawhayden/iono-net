import flax.linen as nn
import jax.numpy as jnp

class ConfigurableModel(nn.Module):
    architecture: list
    activation_fn: callable
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, deterministic: bool):
        for layer_size in self.architecture:
            x = nn.Dense(layer_size)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(12)(x)  # Final layer with 12 outputs (real and imaginary parts)
        return x
