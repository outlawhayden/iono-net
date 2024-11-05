from flax import linen as nn
from typing import Callable, Sequence
import jax.numpy as jnp
from jax import random
import jax
import numpy as np

def MSE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact))

def relative_error2(pred,exact):
    return np.linalg.norm(exact-pred,2)/np.linalg.norm(exact,2)

def glorot_normal(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = glorot_stddev * jnp.array(np.random.normal(size=(in_dim, out_dim)))
    return W

def MLP_init_params(layers):
    params = []
    for i in range(len(layers)-1):
        in_dim, out_dim = layers[i], layers[i + 1]
        W = glorot_normal(in_dim, out_dim)
        b = jnp.zeros(out_dim)
        params.append({"W": W, "b": b})
    return params

def MLP(params,inputs,activation):
    for layer in params[:-1]:
        inputs = activation(inputs @ layer["W"] + layer["b"]) 
    W = params[-1]["W"]
    b = params[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    return outputs