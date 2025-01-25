import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

from Helper import *
from Image import *
from Psi import *
import json
import pandas as pd
import matplotlib as mpl
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import jax

mpl.use("Agg")
plt.rcParams.update({'font.size': 22})


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the target directory (`/home/houtlaw/iono-net/model`)
model_dir = os.path.abspath(os.path.join(current_dir, "../../../../model"))

# Add the model directory to sys.path if it's not already there
if model_dir not in sys.path:
    sys.path.append(model_dir)

print(f"Added {model_dir} to sys.path")

# Import the model module
from model import ConfigurableModel

sample_idx = 30


# convert matlab storing of complex numbers to python complex numbers
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    else:
        return complex(s.replace('i', 'j'))
    

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)
    
###########


# get x axis values
x_range = pd.read_csv('/home/houtlaw/iono-net/data/baselines/high_everything/meta_X_20250124_085447.csv').iloc[:,0].values

# get config dictionary
with open('/home/houtlaw/iono-net/data/baselines/high_everything/setup_20250124_085447.json') as f:
    setup = json.load(f)

print(setup)

F = setup["F"]
ionoNHarm = setup["ionoNharm"]
xi = setup["xi"]
windowType = setup["windowType"]
sumType = setup["sumType"]
#dx = setup["dx"]
dx = 0.25


# get true point scatterer (with relnoise)
scatterer_path_relnoise = '/home/houtlaw/iono-net/data/baselines/high_everything/train_nuStruct_withSpeckle_20250124_085442.csv'
true_scatterers_relnoise = pd.read_csv(scatterer_path_relnoise).map(convert_to_complex).iloc[:,sample_idx].map(np.absolute).values
fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, true_scatterers_relnoise)
plt.title("True Point Scatterers (with Noise)")
plt.savefig('scatterers.png', dpi=300)




# get noisy signal
signal_path_relnoise= "/home/houtlaw/iono-net/data/baselines/high_everything/test_uscStruct_vals_20250124_085447.csv"
signal_df_relnoise = pd.read_csv(signal_path_relnoise).map(convert_to_complex).T.iloc[sample_idx,:].values
sample_signal_vals_relnoise= np.vstack((x_range, signal_df_relnoise))

# get kpsi values
kpsi_path = "/home/houtlaw/iono-net/data/baselines/high_everything/kPsi_20250124_085447.csv"
kpsi_df = pd.read_csv(kpsi_path)

kpsi_values = kpsi_df.values
print("KPsi Values:", kpsi_values)



# get psi coefficients
psi_coeffs_path_relnoise = "/home/houtlaw/iono-net/data/baselines/high_everything/test_compl_ampls_20250124_085446.csv"
psi_coeffs_df_relnoise = pd.read_csv(psi_coeffs_path_relnoise).T
for col in psi_coeffs_df_relnoise.columns:
    # Replace 'i' with 'j' for Python's complex number format and convert to complex numbers
    psi_coeffs_df_relnoise[col] = psi_coeffs_df_relnoise[col].str.replace('i', 'j').apply(complex)

psi_coeffs_vals = psi_coeffs_df_relnoise.iloc[sample_idx,:].values
print("Psi Coefficient Values:", psi_coeffs_vals)

sin_coeffs = []
cos_coeffs = []

for j in psi_coeffs_vals:
    cos_coeffs.append(j.real)
    sin_coeffs.append(-j.imag)




fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, np.absolute(sample_signal_vals_relnoise[1,:]))
plt.title("Sample Input Signal")
plt.savefig('input_signal.png', dpi=300)

rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
rec_fourier_psi.cache_psi(x_range, F, dx, xi)

image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals_relnoise, psi_obj=rec_fourier_psi, F=F)

image_integral = image_object._evaluate_image()

fig = plt.figure(figsize=(30, 8))

plt.plot(x_range, true_scatterers_relnoise, 'orange', lw=3)
plt.plot(x_range, np.absolute(image_integral)/dx,lw = 2)
plt.title("Image Integral")
plt.legend(["True Point Scatterers", "Image Integral"])
plt.savefig('image_integral.png', dpi=300)


print("Loading Baseline Model...")

# get model output coefficients
with open('/home/houtlaw/iono-net/data/baselines/high_everything/model_weights_20250124_094726.pkl', 'rb') as f:
    params = pickle.load(f)

# Define the model with the same architecture as used in training
#architecture = [3000,1093, 832, 328, 963,250, 188, 514]
architecture = [1093,328,963,188,514] 
activation_fn = jax.numpy.tanh  # Load from config if required
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)


# Run inference
def run_inference(model, params, input_data):
    output = model.apply({'params': params}, input_data, deterministic=True)
    return output

model_output = run_inference(model, params, split_complex_to_imaginary(sample_signal_vals_relnoise[1,:]))

# formatted list of psi complex coefficients
model_output_complex = model_output[:len(model_output)//2] + 1j*model_output[len(model_output)//2:]


print("NN Psi Coeffs:", model_output_complex)  

model_sin_coeffs = []
model_cos_coeffs = []

for j in model_output_complex:
    model_cos_coeffs.append(j.real)
    model_sin_coeffs.append(-j.imag)





model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
model_rec_fourier_psi.cache_psi(x_range, F, dx, xi)

model_image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals_relnoise, psi_obj=model_rec_fourier_psi, F=F)

model_image_integral = model_image_object._evaluate_image()



fig = plt.figure(figsize=(30, 8))
plt.rcParams["figure.figsize"] = (100,100)
plt.plot(x_range, true_scatterers_relnoise, 'orange', lw=3)
plt.plot(x_range, np.absolute(model_image_integral)/dx,lw = 2)
plt.title("Image Integral (NN) Inference on Higher Noise Data")
plt.legend(["True Point Scatterers", "Image Integral  (NN)"])
plt.savefig('neural_image_integral_high_params.png', dpi=300)


fig = plt.figure(figsize=(30, 8))
plt.rcParams["figure.figsize"] = (100,100)
plt.plot(x_range, np.absolute(image_integral)/dx, 'red', lw = 6)
plt.plot(x_range, np.absolute(model_image_integral)/dx, 'blue', lw = 1)
plt.title("Image Integral (NN) Inference")
plt.legend([ "Image Integral (Known)", "Image Integral (NN)"])
plt.savefig('neural_image_integral_combined_high_params.png', dpi=300)


# compare literal psi value reconstruction

def buildPsiVals(kPsi, compl_ampls, x):
    val = np.zeros_like(x_range, dtype=float)  # Initialize val as a zero array with the same shape as x
    for ik in range(len(compl_ampls)):
        val += np.real(compl_ampls[ik] * np.exp(1j * kPsi[ik] * x))
    return val

true_psiVals = buildPsiVals(kpsi_values, psi_coeffs_vals, x_range)
nn_psi_vals = buildPsiVals(kpsi_values, model_output_complex, x_range)

# Build Psi values for plotting
model_psiVals = buildPsiVals(kpsi_values, model_output_complex, x_range)

# Plot Psi values
plt.figure(figsize=(10, 6))
plt.rcParams["figure.figsize"] = (100,100)
plt.plot(x_range, true_psiVals, label= "True Psi Values", lw=4)
plt.plot(x_range, model_psiVals, label="Network Psi Values")
plt.xlabel("x")
plt.ylabel("Psi")
plt.title("Psi Reconstruction")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('psi_comparison_high_params.png', dpi=300)



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



# Compute ISLR average
islr_avg = compute_islr(np.absolute(image_integral)/dx, true_scatterers_relnoise, x_range, 5, dx)
print(islr_avg)


islr_samples = []
for i in range(10):
    
    sample_idx = np.random.randint(0, 100)

    signal_df_relnoise = pd.read_csv(signal_path_relnoise).map(convert_to_complex).T.iloc[sample_idx,:].values
    sample_signal_vals_relnoise= np.vstack((x_range, signal_df_relnoise))
    model_output = run_inference(model, params, split_complex_to_imaginary(sample_signal_vals_relnoise[1,:]))
    model_output_complex = model_output[:len(model_output)//2] + 1j*model_output[len(model_output)//2:]

    model_sin_coeffs = []
    model_cos_coeffs = []

    for j in model_output_complex:
        model_cos_coeffs.append(j.real)
        model_sin_coeffs.append(-j.imag)
    
    model_rec_fourier_psi = RecFourierPsi(model_cos_coeffs, model_sin_coeffs, kpsi_values, ionoNHarm)
    model_rec_fourier_psi.cache_psi(x_range, F, dx, xi)

    model_image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals_relnoise, psi_obj=model_rec_fourier_psi, F=F)

    model_image_integral = model_image_object._evaluate_image()

    true_scatterers_relnoise = pd.read_csv(scatterer_path_relnoise).map(convert_to_complex).iloc[:,sample_idx].map(np.absolute).values

    islr_samples = np.append(islr_samples, compute_islr(np.absolute(model_image_integral)/dx, true_scatterers_relnoise, x_range, 5, dx))
    print(islr_samples)

    
print("ISLR Average:", np.mean(islr_samples))
print("ISLR Std Dev:", np.std(islr_samples))


