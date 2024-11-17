import numpy as np
from scipy.integrate import trapezoid


def evaluate_image(params):
    """
    Compute the image integral for all points using the given domain.

    Args:
        params (dict): A dictionary containing the following keys:
            - 'domain' (numpy.ndarray): The spatial domain for evaluation.
            - 'signal' (numpy.ndarray): Complex-valued signal with shape (2, N).
            - 'F' (float): Parameter related to the aperture or smoothing window.
            - 'dx' (float): Sampling interval for the domain.
            - 'window_type' (str): The type of window function to use. Options: 'rect', 'parab', 'parab_scaled'.
            - 'psi_type' (str): The type of psi function to use. Options: 'linear', 'fourier', 'random_fourier'.
            - 'psi_args' (dict): Additional arguments for the psi function.

    Returns:
        numpy.ndarray: A 2D array with the domain and computed image values.
    """
    # Extract parameters
    domain = params['domain']
    signal = params['signal']
    F = params['F']
    dx = params['dx']
    window_type = params['window_type']
    psi_type = params['psi_type']
    psi_args = params.get('psi_args', {})

    # Define window functions
    def rect_window(arr):
        return np.ones_like(arr)

    def parab_window(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        return 1 - ((2 * (arr - arr_min)) / (arr_max - arr_min) - 1) ** 2

    def parab_scaled_window(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        window = 1 - ((2 * (arr - arr_min)) / (arr_max - arr_min) - 1) ** 2
        return window / np.mean(window)

    # Map window types to functions
    window_functions = {
        'rect': rect_window,
        'parab': parab_window,
        'parab_scaled': parab_scaled_window,
    }
    window_func = window_functions[window_type]

    # Define psi functions
    def linear_psi(y, slope=1.0, intercept=0.0):
        return slope * y + intercept

    def fourier_psi(y, amps, wavenums):
        phases = np.outer(y, wavenums)
        return np.real(np.sum(np.exp(1j * phases) * amps, axis=1))

    def random_fourier_psi(y, amps, wavenums, seed=55):
        np.random.seed(seed)
        random_phases = np.random.uniform(0, 2 * np.pi, len(wavenums))
        phases = np.outer(y, wavenums) + random_phases
        return np.real(np.sum(np.exp(1j * phases) * amps, axis=1))

    # Map psi types to functions
    psi_functions = {
        'linear': linear_psi,
        'fourier': fourier_psi,
        'random_fourier': random_fourier_psi,
    }
    psi_func = psi_functions[psi_type]

    # Calculate the image
    imag_val = np.empty_like(signal[1, :], dtype='complex128')
    imag_val[:] = np.nan

    for yidx, y in enumerate(domain):
        x0 = max(signal[0, 0], y - F / 2)
        x1 = min(signal[0, -1], y + F / 2)
        mask = (signal[0, :] >= x0) & (signal[0, :] <= x1)

        base = signal[0, mask]
        signal_vals = signal[1, mask]
        waveform = np.exp(-1j * np.pi * (base - y) ** 2 / F)
        psi_vals = np.exp(1j * psi_func(y, **psi_args))
        window = window_func(base)

        heights = waveform * signal_vals * window * psi_vals
        imag_val[yidx] = trapezoid(heights, base, dx) / F

    return np.vstack((domain, imag_val))
