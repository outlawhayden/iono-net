import numpy as np

def compute_islr(image_integral, known_scatterers, x_vals, min_rad, dx):
    islrs = []
    peaks = [x_vals[i] for i, scatterer in enumerate(known_scatterers) if scatterer > 2]

    image_integral = image_integral **2
    # Compute the total integral over the entire range
    total_integral = np.trapz(image_integral, x=x_vals, dx=dx)

    for peak in peaks:
        # Find indices for inner and outer bounds consistently
        inner_indices = [i for i, x in enumerate(x_vals) if np.abs(x - peak) <= min_rad]
        outer_indices = [i for i, x in enumerate(x_vals) if np.abs(x - peak) > min_rad]
        
        # Use indices to truncate x_vals and image_integral
        inner_peak_bounds = x_vals[inner_indices]
        inner_truncated_image_integral = image_integral[inner_indices]
        outer_peak_bounds = x_vals[outer_indices]
        outer_truncated_image_integral = image_integral[outer_indices]
        
        # Compute the peak integral
        peak_integral = np.trapz(inner_truncated_image_integral, x=inner_peak_bounds, dx=dx)

        print(f"Peak Integral: {peak_integral}")
        
        # Compute the remaining integral
        remaining_integral = np.trapz(outer_truncated_image_integral, x=outer_peak_bounds, dx=dx)
        
        # Calculate the ISLR
        if peak_integral == 0.0:
            islr = 0.0  # Avoiding NaNs
        else:  
            islr = 10 * np.log10((remaining_integral) / peak_integral)
        islrs.append(islr)
    
    return np.mean(islrs)