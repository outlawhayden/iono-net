''' ------------------------------------------------------------------
    Modules & Packages
'''
import numpy as np
from functools import lru_cache
from scipy.integrate import trapezoid
# from Psi import StoredPsi, RecFourierPsi
from WindowFuncs import *
from scipy.signal import find_peaks
from math import sqrt

''' ------------------------------------------------------------------
    Utility Helper Functions
'''
def calc_npoints(a, b, dx):
    """Calculate the number of points to achieve dx spacing."""
    N = (b - a) / dx + 1
    if N.is_integer():
        return int(N)
    else:
        raise ValueError(f'Unable to produce integer discretizations for [{a}, {b}] and dx = {dx}')

def get_index(arr, vals, dx):
    """Vectorized version of get_index to handle an array of values."""
    if not arr.size:
        raise ValueError("Input array is empty.")

    # Calculate indices based on equal spacing
    min_val = arr[0]
    indices = np.rint(np.real(vals - min_val) / np.real(dx)).astype(int)

    # Bounds checking
    if np.any((indices < 0) | (indices >= arr.size)):
        raise ValueError("Some calculated indices are outside the valid range.")

    # Ensure the values at the calculated indices match the target values
    if not np.all(arr[indices] == vals):
        raise ValueError("Some values do not match at the calculated indices.")

    return indices


def zoom_in(xBounds, matrix):
    # Find and return a sub-matrix given the bounds
    lower, upper = xBounds
    
    # Ensure the bounds are within the actual bounds of the matrix's first row
    lower_bound = max(lower, np.min(np.real(matrix[0, :])))
    upper_bound = min(upper, np.max(np.real(matrix[0, :])))
    
    # Create a mask that selects columns within the bounds
    mask = (np.real(matrix[0, :]) >= lower_bound) & (np.real(matrix[0, :]) <= upper_bound)
    
    # Apply the mask to get the sub-matrix
    return matrix[:, mask]

def zoom_peaks(matrix, radius):
    # Find the peaks of the signal and return a list of subsets about radius around the peaks
    
    height_range = np.array([0.9, 1.1])
    peak_indices, _ = find_peaks(np.abs(matrix[1, :]), height=height_range)
    
    peak_subsets = []
    dx = np.real(matrix[0, 1] - matrix[0, 0])
    for idx in peak_indices:
        low_idx, high_idx = int(idx - radius / dx), int(idx + radius / dx) + 1
        peak_subsets.append(matrix[:, low_idx:high_idx])
        
    return peak_subsets

def zoom_locs(matrix, lobe_locs, radius):
    
    zoom_subset = []
    dx = np.real(matrix[0, 1] - matrix[0, 0])
    for loc in lobe_locs:
        idx = np.argmin(np.abs(np.real(matrix[0, :]) - loc))
        low_idx, high_idx = int(idx - radius / dx), int(idx + radius / dx) + 1
        zoom_subset.append(matrix[:, low_idx:high_idx])
        
    return zoom_subset    

''' ------------------------------------------------------------------
    Helper Functions for Calculating the Target Scatters
'''

def build_Pure_Target(target_span, target_dict, dx):
    a, b = target_span
    Ndx = calc_npoints(a, b, dx)
    target_span = np.linspace(a, b, Ndx, dtype='complex128')
    target_vals = np.zeros_like(target_span, dtype='complex128')
    
    # Loop through the target locations (which must match a point in the
    # discretization of the the target span) and add a signal value
    for targ_loc, targ_strength in target_dict.items():
        targ_idx = get_index(target_span, targ_loc, dx)
        target_vals[targ_idx] = np.complex128(targ_strength)
    
    return np.vstack((target_span, target_vals))

def add_Clutter(mu_tru, a_clutter, dx = 0.25, seed = 1):
    clutter_mean, clutter_std = (0, 1/sqrt(2))
    np.random.seed(seed)
    real_normal = np.random.normal(clutter_mean, clutter_std, np.size(mu_tru[1, :]))
    np.random.seed(seed + 1)
    imag_normal = np.random.normal(clutter_mean, clutter_std, np.size(mu_tru[1, :]))
    normal_clutter = real_normal + 1j * imag_normal
    clutter_vec = a_clutter * np.sqrt(dx / 2) * normal_clutter
    return np.vstack((mu_tru[0, :], mu_tru[1, :] + clutter_vec))



''' ------------------------------------------------------------------
    Calculating the Signal
''' 
def signal_Integrand(zarr, x, psi_func, mu_arr, F):
    waveform = np.exp(1j * np.pi / F * (x - zarr)**2)
    perturb = np.exp(-1j * psi_func.calc_psi_cache(x))
    return waveform * perturb * mu_arr
    
def evaluate_Signal_Integral(mu, psi_func, F, dx):
    sig_span = mu[0, :]
    sig_vals = np.empty_like(mu[0, :], dtype='complex128')
    sig_vals[:] = np.nan
    
    for xidx, x in enumerate(sig_span):
        z0 = max(mu[0, 0], x - F/2)
        z0idx = get_index(mu[0, :], z0, dx)
        z1 = min(mu[0, -1], x + F/2)
        z1idx = get_index(mu[0, :], z1, dx)
        
        mu_val_arr = mu[1, z0idx : z1idx + 1] # Target Clutter Values
        base = mu[0, z0idx : z1idx + 1] # z values integrating over
        heights = signal_Integrand(base, x, psi_func, mu_val_arr, F)
        sig_vals[xidx] = trapezoid(heights, base, dx)
    return np.vstack((sig_span, sig_vals))

def add_Noise(signal_tru, a_noise, seed = 7):
    noise_mean, noise_std = (0, 1/sqrt(2))
    np.random.seed(seed)
    real_normal = np.random.normal(noise_mean, noise_std, np.size(signal_tru[1, :]))
    np.random.seed(seed + 1)
    imag_normal = np.random.normal(noise_mean, noise_std, np.size(signal_tru[1, :]))
    normal_noise = real_normal + 1j * imag_normal
    noise_vec = a_noise * np.sqrt(1 / 2) * np.max(np.abs(signal_tru[1, :])) * normal_noise
    return np.vstack((signal_tru[0, :], signal_tru[1, :] + noise_vec))          
   


''' ------------------------------------------------------------------
    Calculating the Image
'''
def image_Integrand(xarr, y, psi_func, sig_arr, F, include_rec = True, window_func = rect_window):
    waveform = np.exp(-1j * np.pi * (xarr - y)**2 / F)
    if include_rec:
        psi_rec = np.exp(1j * psi_func.calc_psi_cache(y))
        return waveform * psi_rec * sig_arr * window_func(xarr)
    else:
        return waveform * sig_arr * window_func(xarr)

def evaluate_Image_Integral(signal, psi_func, F, dx, include_rec = True, window_func = rect_window):
    imag_span = signal[0, :]
    imag_val = np.empty_like(signal[1, :], dtype='complex128')
    imag_val[:] = np.nan

    for yidx, y in enumerate(imag_span):
        x0 = max(np.real(signal[0, 0]), y - F/2)
        x0idx = get_index(np.real(signal[0, :]), x0, dx)
        x1 = min(np.real(signal[0, -1]), y + F/2)
        x1idx = get_index(np.real(signal[0, :]), x1, dx)
        
        signal_vals = signal[1, x0idx : x1idx + 1]
        base = signal[0, x0idx : x1idx + 1]
        heights = image_Integrand(base, y, psi_func, signal_vals, F, include_rec, window_func)
        imag_calc = trapezoid(heights, base, dx)
        imag_val[yidx] = imag_calc


