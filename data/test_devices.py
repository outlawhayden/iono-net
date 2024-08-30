import jax.numpy as jnp
import numpy as np
import jax as jax
print("Backend Selected:", jax.lib.xla_bridge.get_backend().platform)
print("Detected Devices:", jax.devices())

