'''-------------------------------------------------------------  
    Window Functions Module:
    
    This module provides various window functions that can be applied to arrays.
    These functions are used to shape the sampling window, commonly in signal processing or similar applications.
    
    Included Window Functions:
    - Parabolic (Scaled): A parabolic window that is scaled to have an area of 1.
    - Rectangular: A simple window where all values within the window are set to 1.
'''

import numpy as np

def rect_window(arr):
    """
    Apply a rectangular window function to an array.
    
    Args:
        arr (numpy.ndarray): The input array to which the window will be applied.
        
    Returns:
        numpy.ndarray: An array of ones with the same shape as the input array.
    """
    return np.ones_like(arr)

def parab_window(arr):
    """
    Apply a parabolic window function to an array.
    
    Args:
        arr (numpy.ndarray): The input array to which the window will be applied.
        
    Returns:
        numpy.ndarray: An array where each element is calculated using a parabolic function centered within the array bounds.
    """
    arr_min, arr_max = np.min(arr), np.max(arr)
    return 1 - ((2 * (arr - arr_min)) / (arr_max - arr_min) - 1) ** 2

def parab_scaled_window(arr):
    """
    Apply a scaled parabolic window function to an array where the window's area sums to 1.
    
    Args:
        arr (numpy.ndarray): The input array to which the window will be applied.
        
    Returns:
        numpy.ndarray: An array where each element is calculated using a parabolic function scaled such that the mean value is 1.
    """
    arr_min, arr_max = np.min(arr), np.max(arr)
    window_arr = 1 - ((2 * (arr - arr_min)) / (arr_max - arr_min) - 1) ** 2
    return window_arr / np.mean(window_arr)
