import flax.linen as nn
import jax.numpy as jnp

class ComplexDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        # x: complex-valued tensor of shape (batch, in_features)
        real = jnp.real(x)
        imag = jnp.imag(x)

        W_real = nn.Dense(self.features, name="W_real")
        W_imag = nn.Dense(self.features, name="W_imag")

        out_real = W_real(real) - W_imag(imag)
        out_imag = W_real(imag) + W_imag(real)

        return out_real + 1j * out_imag

def complex_relu(z):
    return jnp.maximum(jnp.real(z), 0) + 1j * jnp.maximum(jnp.imag(z), 0)


class ComplexMLP(nn.Module):
    architecture: list
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        if x.ndim == 3:  # (batch, length, 2)
            x = x[..., 0] + 1j * x[..., 1]  # convert to complex (batch, length)
        x = x.reshape((x.shape[0], -1))  # flatten

        for layer_size in self.architecture:
            x = ComplexDense(layer_size)(x)
            x = complex_relu(x)
            # Dropout: approximate with real dropout on magnitude
            mag = jnp.abs(x)
            mag = jnp.where(mag == 0, 1e-8, mag)
            mask = nn.Dropout(rate=self.dropout_rate)(mag, deterministic=deterministic)
            phase = jnp.angle(x)
            x = mask * jnp.exp(1j * phase)

        x = ComplexDense(6)(x)  # Output: 6 complex coefficients
        return x
