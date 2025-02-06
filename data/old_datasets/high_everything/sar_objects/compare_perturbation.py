import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
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
model_dir = os.path.abspath('model')

# Add the model directory to sys.path if it's not already there
if model_dir not in sys.path:
    sys.path.insert(0, '/home/houtlaw/iono-net/model')

print(f"Added {model_dir} to sys.path")

# Import the model module
from model import ConfigurableModel

sample_idx = 1


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
x_range = pd.read_csv('/home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/meta_X_20250107_122642.csv').iloc[:,0].values

# get config dictionary
with open('/home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/setup_20250107_122642.json') as f:
    setup = json.load(f)

print(setup)

F = setup["F"]
ionoNHarm = setup["ionoNharm"]
xi = setup["xi"]
windowType = setup["windowType"]
sumType = setup["sumType"]
#dx = setup["dx"]
dx = 0.25


# get true point scatterer (with perturbation)
scatterer_path_perturb = '/home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/nuStruct_withSpeckle_20250107_122640.csv'
true_scatterers_perturb = pd.read_csv(scatterer_path_perturb).map(convert_to_complex).iloc[:,sample_idx].map(np.absolute).values
fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, true_scatterers_perturb)
plt.title("True Point Scatterers (more Perturbation)")
plt.savefig('scatterers.png', dpi=300)




# get noisy signal
signal_path_perturb = "//home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/uscStruct_vals_20250107_122641.csv"
signal_df_perturb = pd.read_csv(signal_path_perturb).map(convert_to_complex).T.iloc[sample_idx,:].values
sample_signal_vals_perturb= np.vstack((x_range, signal_df_perturb))

# get kpsi values
kpsi_path = "/home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/kPsi_20250107_122642.csv"
kpsi_df = pd.read_csv(kpsi_path)

kpsi_values = kpsi_df.values
print("KPsi Values:", kpsi_values)



# get psi coefficients
psi_coeffs_path_perturb = "/home/houtlaw/iono-net/utils/perturbation_experiments/more_perturb/compl_ampls_20250107_122641.csv"
psi_coeffs_df_perturb = pd.read_csv(psi_coeffs_path_perturb).T
for col in psi_coeffs_df_perturb.columns:
    # Replace 'i' with 'j' for Python's complex number format and convert to complex numbers
    psi_coeffs_df_perturb[col] = psi_coeffs_df_perturb[col].str.replace('i', 'j').apply(complex)

psi_coeffs_vals = psi_coeffs_df_perturb.iloc[sample_idx,:].values
print("Psi Coefficient Values:", psi_coeffs_vals)

sin_coeffs = []
cos_coeffs = []

for j in psi_coeffs_vals:
    cos_coeffs.append(j.real)
    sin_coeffs.append(-j.imag)




fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, np.absolute(sample_signal_vals_perturb[1,:]))
plt.title("Sample Input Signal (More Perturbation)")
plt.savefig('input_signal.png', dpi=300)

rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
rec_fourier_psi.cache_psi(x_range, F, dx, xi)

image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals_perturb, psi_obj=rec_fourier_psi, F=F)

image_integral = image_object._evaluate_image()

fig = plt.figure(figsize=(30, 8))

plt.plot(x_range, true_scatterers_perturb, 'orange', lw=3)
plt.plot(x_range, np.absolute(image_integral)/dx,lw = 2)
plt.title("Image Integral (More Perturbation)")
plt.legend(["True Point Scatterers", "Image Integral"])
plt.savefig('image_integral.png', dpi=300)


print("Loading Baseline Model...")

# get model output coefficients
with open('/home/houtlaw/iono-net/utils/perturbation_experiments/baseline/model_weights_20250107_130653.pkl', 'rb') as f:
    params = pickle.load(f)

# Define the model with the same architecture as used in training
architecture = [1093,328,963,188,514] 
activation_fn = jax.numpy.tanh  # Load from config if required
model = ConfigurableModel(architecture=architecture, activation_fn=activation_fn)


# Run inference
def run_inference(model, params, input_data):
    output = model.apply({'params': params}, input_data, deterministic=True)
    return output

model_output = run_inference(model, params, split_complex_to_imaginary(sample_signal_vals_perturb[1,:]))

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

model_image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals_perturb, psi_obj=model_rec_fourier_psi, F=F)

model_image_integral = model_image_object._evaluate_image()



fig = plt.figure(figsize=(30, 8))
plt.rcParams["figure.figsize"] = (100,100)
plt.plot(x_range, true_scatterers_perturb, 'orange', lw=3)
plt.plot(x_range, np.absolute(model_image_integral)/dx,lw = 2)
plt.title("Image Integral (NN) Inference on Higher Perturbation Data")
plt.legend(["True Point Scatterers (Higher Perturbation)", "Image Integral  (NN)"])
plt.savefig('neural_image_integral_more_perturb.png', dpi=300)


fig = plt.figure(figsize=(30, 8))
plt.rcParams["figure.figsize"] = (100,100)
plt.plot(x_range, np.absolute(image_integral)/dx, 'red', lw = 6)
plt.plot(x_range, np.absolute(model_image_integral)/dx, 'blue', lw = 1)
plt.title("Image Integral (NN) Inference on Higher Perturbation Data")
plt.legend([ "Image Integral (Known)", "Image Integral (NN)"])
plt.savefig('neural_image_integral_combined_more_perturb.png', dpi=300)


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
plt.title("Psi Reconstruction (Trained on Lower Perturb)")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('psi_comparison_more_perturb.png', dpi=300)

