import numpy as np
from multiprocessing import cpu_count
from scipy.optimize import minimize
from math import floor
from scipy.stats.qmc import LatinHypercube
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
from time import time

'''----------------------------------------------------------------------------------
    Cost Function & Gradient Calculation
'''

def calc_Cost(ampl_guess, psi_func, image, zeta):
    """
    Calculate the combined cost function incorporating image processing and regularization.
    
    Args:
        ampl_guess (np.array): Current guess for the amplitudes.
        psi_func (object): Psi function instance handling phase perturbations.
        image (object): Image instance to be optimized.
        zeta (float): Regularization parameter.
    
    Returns:
        float: The calculated cost for the given amplitude guess.
    """
    psi_func.update_amps(
        ampl_guess[:psi_func.Nharmonics],
        ampl_guess[psi_func.Nharmonics:2 * psi_func.Nharmonics]
    )
    image.update_image(psi_func)
    cost = -image.dx * image.calc_cost() + zeta * psi_func.calc_cost()
    return cost

def calc_Grad(_, psi_func, image, zeta):
    """
    Calculate the gradient of the cost function.
    
    Args:
        _ (ignored): Placeholder for compatibility with optimization routines.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        np.array: Gradient of the cost function.
    """
    return np.real(-image.dx * image.calc_grad() + zeta * psi_func.calc_grad())

'''----------------------------------------------------------------------------------------
    Optimizer Helper Functions
'''

def optim_at_bound(optim_amps, bounds):
    """
    Check if the optimization amplitudes are close to the bounds.
    
    Args:
        optim_amps (np.array): Amplitudes from the optimizer.
        bounds (list of tuples): Lower and upper bounds for each amplitude.
    
    Returns:
        bool: True if any amplitude is close to its bounds, otherwise False.
    """
    eps = 1e-8
    optim_amps = np.asarray(optim_amps)
    bounds = np.asarray(bounds)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    
    return np.any((optim_amps + eps >= upper_bounds) | (optim_amps - eps <= lower_bounds))

'''----------------------------------------------------------------------------------------
    Local Optimization
'''

def local(ampl_guess, psi_func, image, zeta):
    """
    Perform local optimization starting from a given amplitude guess.
    
    Args:
        ampl_guess (np.array): Initial guess for the amplitudes.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        dict: Dictionary containing optimization results and metadata.
    """
    t0 = time()
    results = minimize(
        fun=calc_Cost, x0=ampl_guess, args=(psi_func, image, zeta), method='BFGS',
        jac=calc_Grad, options={'maxiter': 150, 'gtol': 0.001}
    )
    t1 = time()
    
    results_dict = {
        'x': results.x, 'fun': results.fun, 'init': ampl_guess, 'time': t1 - t0, 'points': 1
    }
    return results_dict

'''------------------------------------------------------------------------------------
    Global Optimization: Multi-Start Strategy
'''

def multistart(lead_amp_max, decay_rate, workers, psi_func, image, zeta):
    """
    Multi-start optimization using Latin Hypercube Sampling for initial points.
    
    Args:
        lead_amp_max (float): Maximum amplitude for the leading harmonic.
        decay_rate (float): Decay rate for subsequent harmonics.
        workers (int): Number of parallel workers.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        tuple: Contains the best global result and a DataFrame of all results.
    """
    N = psi_func.Nharmonics
    upper = lead_amp_max * decay_rate ** np.arange(N)
    bounds = np.array(list(zip(-upper, upper)))

    sampler = LatinHypercube(2 * N)
    init_points = sampler.random(workers) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    t0 = time()
    with ProcessPoolExecutor(max_workers=min(workers, cpu_count())) as executor:
        optim_w_args = partial(local, psi_func=psi_func, image=image, zeta=zeta)
        results = list(executor.map(optim_w_args, init_points))
    
    results_df = pd.DataFrame({
        'Cost': [r['fun'] for r in results],
        'Init': [r['init'] for r in results],
        'Optim': [r['x'] for r in results]
    })
    best_result = min(results, key=lambda x: x['fun'])
    best_result.update({'time': time() - t0, 'points': workers})
    
    return best_result, results_df

import numpy as np
from multiprocessing import cpu_count
from scipy.optimize import minimize
from math import floor
from scipy.stats.qmc import LatinHypercube
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
from time import time

'''----------------------------------------------------------------------------------
    Cost Function & Gradient Calculation
'''

def calc_Cost(ampl_guess, psi_func, image, zeta):
    """
    Calculate the combined cost function incorporating image processing and regularization.
    
    Args:
        ampl_guess (np.array): Current guess for the amplitudes.
        psi_func (object): Psi function instance handling phase perturbations.
        image (object): Image instance to be optimized.
        zeta (float): Regularization parameter.
    
    Returns:
        float: The calculated cost for the given amplitude guess.
    """
    psi_func.update_amps(
        ampl_guess[:psi_func.Nharmonics],
        ampl_guess[psi_func.Nharmonics:2 * psi_func.Nharmonics]
    )
    image.update_image(psi_func)
    cost = -image.dx * image.calc_cost() + zeta * psi_func.calc_cost()
    return cost

def calc_Grad(_, psi_func, image, zeta):
    """
    Calculate the gradient of the cost function.
    
    Args:
        _ (ignored): Placeholder for compatibility with optimization routines.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        np.array: Gradient of the cost function.
    """
    return np.real(-image.dx * image.calc_grad() + zeta * psi_func.calc_grad())

'''----------------------------------------------------------------------------------------
    Optimizer Helper Functions
'''

def optim_at_bound(optim_amps, bounds):
    """
    Check if the optimization amplitudes are close to the bounds.
    
    Args:
        optim_amps (np.array): Amplitudes from the optimizer.
        bounds (list of tuples): Lower and upper bounds for each amplitude.
    
    Returns:
        bool: True if any amplitude is close to its bounds, otherwise False.
    """
    eps = 1e-8
    optim_amps = np.asarray(optim_amps)
    bounds = np.asarray(bounds)
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    
    return np.any((optim_amps + eps >= upper_bounds) | (optim_amps - eps <= lower_bounds))

'''----------------------------------------------------------------------------------------
    Local Optimization
'''

def local(ampl_guess, psi_func, image, zeta):
    """
    Perform local optimization starting from a given amplitude guess.
    
    Args:
        ampl_guess (np.array): Initial guess for the amplitudes.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        dict: Dictionary containing optimization results and metadata.
    """
    t0 = time()
    results = minimize(
        fun=calc_Cost, x0=ampl_guess, args=(psi_func, image, zeta), method='BFGS',
        jac=calc_Grad, options={'maxiter': 150, 'gtol': 0.001}
    )
    t1 = time()
    
    results_dict = {
        'x': results.x, 'fun': results.fun, 'init': ampl_guess, 'time': t1 - t0, 'points': 1
    }
    return results_dict

'''------------------------------------------------------------------------------------
    Global Optimization: Multi-Start Strategy
'''

def multistart(lead_amp_max, decay_rate, workers, psi_func, image, zeta):
    """
    Multi-start optimization using Latin Hypercube Sampling for initial points.
    
    Args:
        lead_amp_max (float): Maximum amplitude for the leading harmonic.
        decay_rate (float): Decay rate for subsequent harmonics.
        workers (int): Number of parallel workers.
        psi_func (object): Psi function instance.
        image (object): Image instance.
        zeta (float): Regularization parameter.
    
    Returns:
        tuple: Contains the best global result and a DataFrame of all results.
    """
    N = psi_func.Nharmonics
    upper = lead_amp_max * decay_rate ** np.arange(N)
    bounds = np.array(list(zip(-upper, upper)))

    sampler = LatinHypercube(2 * N)
    init_points = sampler.random(workers) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    t0 = time()
    with ProcessPoolExecutor(max_workers=min(workers, cpu_count())) as executor:
        optim_w_args = partial(local, psi_func=psi_func, image=image, zeta=zeta)
        results = list(executor.map(optim_w_args, init_points))
    
    results_df = pd.DataFrame({
        'Cost': [r['fun'] for r in results],
        'Init': [r['init'] for r in results],
        'Optim': [r['x'] for r in results]
    })
    best_result = min(results, key=lambda x: x['fun'])
    best_result.update({'time': time() - t0, 'points': workers})
    
    return best_result, results_df

from functools import lru_cache
from typing import Any
import numpy as np
from scipy.integrate import trapezoid
from Helper import get_index

''' ------------------------------------------------------------------
    Base Psi Class for Ionospheric Phase Perturbation
'''
class Psi:
    def __init__(self, domain=np.arange(0, 360 + 1/8, 1/8), xi=0.5) -> None:
        """Initialize the base Psi class with domain and xi parameters."""
        self.domain = domain
        self.xi = xi
        self.psi_cache = {}

    def calc_sarr_linear(self, x, yORz, xi):
        """Calculate linear combination of positions for psi computation."""
        return self.xi * x + (1 - self.xi) * yORz

''' ------------------------------------------------------------------
    Linear Perturbation Function
'''
class LinearPsi(Psi):
    def __init__(self, slope=1, intercept=0) -> None:
        """Initialize the LinearPsi class with slope and intercept for the psi function."""
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def calc_psi(self, sarr):
        """Calculate psi values using a linear equation."""
        return self.slope * sarr + self.intercept

''' ------------------------------------------------------------------
    Fourier Perturbation Function
'''
class FourierPsi(Psi):
    def __init__(self, amps, wavenums, Nharmonics, seed=55) -> None:
        """Initialize FourierPsi with amplitudes, wavenumbers, and the number of harmonics."""
        super().__init__()
        self.amps = amps
        self.wavenums = wavenums
        self.Nharmonics = Nharmonics
        np.random.seed(seed)
        self.perturbs = np.random.uniform(0, 2 * np.pi, Nharmonics)

    def calc_psi(self, sarr):
        """Calculate psi values using a Fourier series based on random perturbations."""
        phases = np.outer(sarr, self.wavenums) + self.perturbs
        return np.real(np.exp(1j * phases) @ self.amps)

''' ------------------------------------------------------------------
    Reconstructed Fourier Perturbation Function
'''
class RecFourierPsi(Psi):
    def __init__(self, cosAmps, sinAmps, wavenums, Nharmonics) -> None:
        """Initialize RecFourierPsi with cosine and sine amplitude arrays, wavenumbers, and harmonics."""
        super().__init__()
        self.cosAmps = cosAmps
        self.sinAmps = sinAmps
        self.wavenums = wavenums
        self.Nharmonics = Nharmonics
        self.vertShift = 0
        self.cache_y2s = {}  # cache for sarr values
        self.cache_y2psi = {}  # cache for psi values

    def cache_psi(self, domain, F, dx, xi):
        """Cache psi values for a given domain and parameters."""
        for i, y in enumerate(domain):
            xarr = domain[np.maximum(0, i - int(F/2/dx)): i + int(F/2/dx) + 1]
            sarr = self.calc_sarr_linear(xarr, y, xi)
            self.cache_y2s[y] = sarr
            self.cache_y2psi[y] = np.real(np.sum(self.cosAmps * np.cos(np.outer(sarr, self.wavenums)) + 
                                                self.sinAmps * np.sin(np.outer(sarr, self.wavenums)), axis=1))

    def calc_psi_cache(self, y):
        """Retrieve or calculate cached psi values."""
        if y not in self.cache_y2psi:
            self.cache_y2psi[y] = self.calc_psi(self.cache_y2s[y])
        return self.cache_y2psi[y]

    def update_amps(self, cosAmps, sinAmps):
        """Update amplitudes and recalculate the psi cache."""
        self.cosAmps, self.sinAmps = cosAmps, sinAmps
        for y, s_val in self.cache_y2s.items():
            self.cache_y2psi[y] = np.real(np.sum(self.cosAmps * np.cos(np.outer(s_val, self.wavenums)) + 
                                                self.sinAmps * np.sin(np.outer(s_val, self.wavenums)), axis=1))

    def solve_amps(self, xvals, yvals, period, wavenums, Nharmonics):
        """Compute the amplitudes for the Fourier series from the given data."""
        dx = xvals[1] - xvals[0]
        period_idx = int(period / dx) + 1
        a0 = 2 / period * trapezoid(y=yvals[:period_idx], x=xvals[:period_idx])
        self.vertShift = a0
        self.cosAmps = [2 / period * trapezoid(yvals[:period_idx] * np.cos(k * xvals[:period_idx]), dx=dx) for k in wavenums]
        self.sinAmps = [2 / period * trapezoid(yvals[:period_idx] * np.sin(k * xvals[:period_idx]), dx=dx) for k in wavenums]

    def calc_cost(self):
        """Calculate cost as a function of harmonic amplitudes and wavenumbers."""
        return np.sum(self.wavenums ** 2 * (self.cosAmps ** 2 + self.sinAmps ** 2))
    
    def calc_grad(self):
        """Calculate gradient of the cost function."""
        k2 = 2 * self.wavenums ** 2
        cos_grad = k2 * self.cosAmps
        sin_grad = k2 * self.sinAmps
        return np.concatenate((cos_grad, sin_grad))

