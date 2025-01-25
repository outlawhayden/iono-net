import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the ISLR computation function
def compute_islr(image_integral, known_scatterers, x_vals, min_rad, dx):
    islrs = []
    peaks = [x_vals[i] for i, scatterer in enumerate(known_scatterers) if scatterer > 0.5]

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


# Read x_range data
x_range = pd.read_csv('/home/houtlaw/iono-net/data/baselines/high_everything/meta_X_20250124_085447.csv').values.flatten()

# Define scatterer data
scatterer_data = np.zeros_like(x_range)
#scatterer_index = len(scatterer_data) // 2  # Center index
scatterer_index = 0
scatterer_data[scatterer_index] = 1  # Delta function at the center

# Generate sinc data
#sinc_data = np.abs(np.sinc((x_range - x_range[scatterer_index]) / np.pi))  # Shift sinc to center at scatterer
sinc_data = np.abs(np.sinc((x_range) / np.pi) ** 2)

# Plotting
plt.figure(figsize=(20, 6))
plt.plot(x_range, scatterer_data, label="Delta Function (Point Scatterer)", lw=4)
plt.plot(x_range, sinc_data, label="Shifted Sinc Function", lw=4, color='orange')
plt.xlabel("x_range")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Delta and Sinc Function Comparison (Shifted)")
plt.grid(True)
plt.savefig('islr_test.png', dpi=300)
plt.show()

islr_val = compute_islr(sinc_data, scatterer_data, x_range, np.pi, 0.25)

print(islr_val)
