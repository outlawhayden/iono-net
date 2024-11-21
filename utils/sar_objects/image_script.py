from Helper import *
from Image import *
from Psi import *
import json
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.use("Agg")

### VARIABLES
sample_idx = 1


# convert matlab storing of complex numbers to python complex numbers
def convert_to_complex(s):
    if s == "NaNNaNi":
        return np.NaN
    else:
        return complex(s.replace('i', 'j'))
    

# get x axis values
x_range = pd.read_csv('/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/meta_X_20241117_203151.csv').iloc[:,0].values

# get config dictionary
with open('/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/setup_20241117_203151.json') as f:
    setup = json.load(f)

print(setup)

F = setup["F"]
ionoNHarm = setup["ionoNharm"]
xi = setup["xi"]
windowType = setup["windowType"]
sumType = setup["sumType"]
#dx = setup["dx"]
dx = 0.25


# get true point scatterer (with speckle)
scatterer_path = '/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/nuStruct_withSpeckle_20241117_203151.csv'
true_scatterers = pd.read_csv(scatterer_path).map(convert_to_complex).iloc[:,sample_idx].map(np.absolute).values
fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, true_scatterers)
plt.title("True Point Scatterers (with Speckle)")
plt.savefig('scatterers.png', dpi=300)




# get noisy signal
signal_path = "/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/uscStruct_vals_20241117_203151.csv"
signal_df = pd.read_csv(signal_path).map(convert_to_complex).T.iloc[sample_idx,:].values
sample_signal_vals = np.vstack((x_range, signal_df))

# get kpsi values
kpsi_path = "/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/kPsi_20241117_203151.csv"
kpsi_df = pd.read_csv(kpsi_path)

kpsi_values = kpsi_df.values
print("KPsi Values:", kpsi_values)



# get psi coefficients
psi_coeffs_path = "/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/compl_ampls_20241117_203151.csv"
psi_coeffs_df = pd.read_csv(psi_coeffs_path).T
for col in psi_coeffs_df.columns:
    # Replace 'i' with 'j' for Python's complex number format and convert to complex numbers
    psi_coeffs_df[col] = psi_coeffs_df[col].str.replace('i', 'j').apply(complex)

psi_coeffs_vals = psi_coeffs_df.iloc[sample_idx,:].values
print("Psi Coefficient Values:", psi_coeffs_vals)

sin_coeffs = []
cos_coeffs = []

for j in psi_coeffs_vals:
    cos_coeffs.append(j.real)
    sin_coeffs.append(-j.imag)



fig = plt.figure(figsize=(30, 8))
plt.plot(x_range, np.absolute(sample_signal_vals[1,:]))
plt.title("Sample Input Signal")
plt.savefig('input_signal.png', dpi=300)

rec_fourier_psi = RecFourierPsi(cos_coeffs, sin_coeffs, kpsi_values, ionoNHarm)
rec_fourier_psi.cache_psi(x_range, F, dx, xi)

image_object = Image(x_range, window_func=rect_window, signal=sample_signal_vals, psi_obj=rec_fourier_psi, F=F)

image_integral = image_object._evaluate_image()

fig = plt.figure(figsize=(30, 8))

plt.plot(x_range, true_scatterers, 'orange', lw=3)
plt.plot(x_range, np.absolute(image_integral),lw = 2)
plt.title("Image Integral")
plt.savefig('image_integral.png', dpi=300)
