import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional
from flax.linen.initializers import variance_scaling

def small_kernel_init(scale=1e-3):
    return variance_scaling(scale, "fan_avg", "uniform")

# A convolutional block that applies two 1D convolution layers with ReLU activation
class ConvBlock(nn.Module):
    features: int
    kernel_size: int = 11
    negative_slope: float = 0.001
    kernel_init: callable = small_kernel_init()

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(self.kernel_size,), padding='SAME',
                    kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = nn.leaky_relu(x, self.negative_slope)
        x = nn.Conv(self.features, kernel_size=(self.kernel_size,), padding='SAME',
                    kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = nn.leaky_relu(x, self.negative_slope)
        return x

class DownBlock(nn.Module):
    features: int
    kernel_init: callable = small_kernel_init()

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features, kernel_init=self.kernel_init)(x)
        skip = x
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        return x, skip

class UpBlock(nn.Module):
    out_channels: int
    negative_slope: float = 0
    kernel_init: callable = small_kernel_init()

    @nn.compact
    def __call__(self, x, skip):
        x = nn.ConvTranspose(features=self.out_channels, kernel_size=(2,), strides=(2,),
                             kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)

        diff = skip.shape[1] - x.shape[1]
        if diff > 0:
            skip = skip[:, :-diff, :]
        elif diff < 0:
            x = x[:, :-abs(diff), :]

        x = jnp.concatenate([x, skip], axis=-1)
        x = nn.Conv(self.out_channels, kernel_size=(3,), padding="SAME",
                    kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = nn.leaky_relu(x, self.negative_slope)
        return x

class UNet1D_i2i(nn.Module):
    down_channels: Sequence[int]
    bottleneck_channels: int
    up_channels: Optional[Sequence[int]]
    output_dim: int = 2
    kernel_init: callable = small_kernel_init()

    @nn.compact
    def __call__(self, x):
        if x.ndim == 2:
            x = x[..., None]

        skips = []
        for ch in self.down_channels:
            x, skip = DownBlock(ch, kernel_init=self.kernel_init)(x)
            skips.append(skip)

        x = ConvBlock(self.bottleneck_channels, kernel_init=self.kernel_init)(x)

        if self.up_channels is not None and len(self.up_channels) > 0:
            for ch, skip in zip(self.up_channels, reversed(skips)):
                x = UpBlock(ch, kernel_init=self.kernel_init)(x, skip)

        x = nn.Conv(self.output_dim, kernel_size=(1,), padding='SAME',
                    kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        return x
