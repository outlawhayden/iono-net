import jax
import jax.numpy as jnp
import flax.linen as nn

# Complex convolution using real-valued layers over 2-channel inputs
class ComplexConv1D(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        real = nn.Conv(self.features, self.kernel_size, padding="SAME")(x[..., 0:1])
        imag = nn.Conv(self.features, self.kernel_size, padding="SAME")(x[..., 1:2])
        return jnp.concatenate([real, imag], axis=-1)

class ComplexReLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.stack([nn.relu(x[..., 0]), nn.relu(x[..., 1])], axis=-1)

class EncoderBlock(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        x = ComplexConv1D(self.features, self.kernel_size)(x)
        x = ComplexReLU()(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding="SAME")
        return x

class DecoderBlock(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, skip):
        x = jax.image.resize(x, shape=(x.shape[0], skip.shape[1], x.shape[2]), method='linear')
        x = jnp.concatenate([x, skip], axis=-1)
        x = ComplexConv1D(self.features, self.kernel_size)(x)
        x = ComplexReLU()(x)
        return x

class ComplexUNet1D(nn.Module):
    depth: int = 3
    base_features: int = 32
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        skips = []
        for i in range(self.depth):
            x = EncoderBlock(self.base_features * 2**i, self.kernel_size)(x)
            skips.append(x)

        x = ComplexConv1D(self.base_features * 2**self.depth, self.kernel_size)(x)
        x = ComplexReLU()(x)

        for i in reversed(range(self.depth)):
            x = DecoderBlock(self.base_features * 2**i, self.kernel_size)(x, skips[i])

        x = jnp.mean(x, axis=1)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(12)(x)
        return x
