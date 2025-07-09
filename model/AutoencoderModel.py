import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

class HybridAutoencoder(nn.Module):
    up_dims: Sequence[int]
    dense_dims: Sequence[int]
    down_dims: Sequence[int]
    activation_fn: callable
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # Input: (B, L, 2)
        for features in self.up_dims:
            x = nn.ConvTranspose(features=features, kernel_size=(3,), strides=(2,), padding="SAME")(x)
            x = self.activation_fn(x)

        # Flatten for dense layers
        B, L, C = x.shape
        x = x.reshape((B, L * C))

        for dim in self.dense_dims:
            x = nn.Dense(dim)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Project back to spatial shape (example: 64 × 8)
        x = nn.Dense(64 * 8)(x)
        x = x.reshape((B, 64, 8))

        for features in self.down_dims:
            x = nn.Conv(features=features, kernel_size=(3,), strides=(2,), padding="SAME")(x)
            x = self.activation_fn(x)

        # Final projection to output vector
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(12)(x)
        return x
