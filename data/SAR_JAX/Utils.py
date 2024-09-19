import numpy as np
import jax as jnp
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from math import sqrt


'''CALCULATION UTILITIES'''
def calc_npoints_jax(a,b,dx): #equal
    N = (b-a) / dx + 1
    if N.is_integer():
        return int(N)
    else:
        raise ValueError(f'Unable to produce integer discretizations for [{a}, {b}] and dx = {dx}')

def get_index_jax(arr, vals, dx):
    if not arr.size:
        raise ValueError("Input array is empty.")

'''TARGET CONSTRUCTION UTILITIES'''
def build_Pure_Target(target_span, target_dict, dx):
    a, b = target_span
    Ndx = calc_npoints_jax(a, b, dx)
    target_span = jnp.linspace(a, b, Ndx, dtype = 'complex128')
    target_vals = jnp.zeros_like(target_span, dtype = 'complex128')

    for targ_loc, targ_strength in target_dict.items():
        targ_idx = get_index(target_span, targ_loc, dx)
