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