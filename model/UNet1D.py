import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any

class ConvBlock(nn.Module):
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(self.kernel_size,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(self.kernel_size,), padding='SAME')(x)
        x = nn.relu(x)
        return x

class DownBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features)(x)
        skip = x
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        return x, skip

class UpBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, skip):
        x = nn.ConvTranspose(features=self.out_channels, kernel_size=(2,), strides=(2,))(x)

        # Match the skip and x shapes along the length dimension
        diff = skip.shape[1] - x.shape[1]
        if diff > 0:
            skip = skip[:, :-diff, :]
        elif diff < 0:
            x = x[:, :-abs(diff), :]

        x = jnp.concatenate([x, skip], axis=-1)
        x = nn.Conv(self.out_channels, kernel_size=(3,), padding="SAME")(x)
        x = nn.relu(x)
        return x

class UNet1D(nn.Module):
    down_channels: Sequence[int]     # e.g., [32, 64, 128]
    bottleneck_channels: int         # e.g., 256
    up_channels: Sequence[int]       # e.g., [128, 64, 32]
    output_dim: int = 12             # final output size

    @nn.compact
    def __call__(self, x):  # x shape: (batch, 1040)
        x = x[..., None]    # expand to (batch, 1040, 1)

        skips = []

        # Down path
        for ch in self.down_channels:
            x, skip = DownBlock(ch)(x)
            skips.append(skip)

        # Bottleneck
        x = ConvBlock(self.bottleneck_channels)(x)

        # Up path
        for ch, skip in zip(self.up_channels, reversed(skips)):
            x = UpBlock(ch)(x, skip)

        # Global average pooling over spatial dimension
        x = jnp.mean(x, axis=1)  # shape: (batch, features)

        # Final dense output
        x = nn.Dense(self.output_dim)(x)
        return x
