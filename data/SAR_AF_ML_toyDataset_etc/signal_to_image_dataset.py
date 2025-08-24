import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# ================================================================
# CONFIGURATION
# ------------------------------------------------
# Define input and output file paths. These point to metadata,
# signal data, Psi coefficient data, and output CSVs for the
# reconstructed focused and unfocused images.
# ================================================================
DATA_DIR = "/home/houtlaw/iono-net/data/aug25"

# Metadata: spatial sampling range (x-axis points)
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250707_155325.csv"

# Setup JSON (contains system parameters like F, xi, etc.)
SETUP_PATH = f"{DATA_DIR}/setup_20250707_155325.json"

# Test set signals (uscStruct values) – raw time-domain samples
SIGNAL_PATH = f"{DATA_DIR}/test_uscStruct_vals_20250707_155324.csv"

# Psi coefficients corresponding to each test signal
PSI_PATH = f"{DATA_DIR}/test_compl_ampls_20250707_155324.csv"

# Harmonic wavenumber values (kPsi)
KPSI_PATH = f"{DATA_DIR}/kPsi_20250707_155325.csv"

# Outputs: reconstructed images (focused, unfocused), trimmed x-range
OUTPUT_PATH_FOCUSED = f"{DATA_DIR}/test_image_recon_jnp.csv"
OUTPUT_PATH_UNFOCUSED = f"{DATA_DIR}/test_image_recon_jnp_unfocused.csv"
X_TRIM_PATH = f"{DATA_DIR}/x_range_image_recon_jnp.csv"

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def convert_to_complex(s):
    """
    Convert a string entry from CSV into a complex number.
    Handles MATLAB-style "NaNNaNi" placeholder specially.
    """
    s = str(s)
    if s == "NaNNaNi":
        return np.nan  # Represent missing data as NaN
    return complex(s.replace("i", "j"))  # MATLAB uses 'i', Python expects 'j'

def compute_image_integral_torch(x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5):
    """
    Compute the image-domain reconstruction integral in PyTorch.

    Inputs:
        x_range (ndarray): spatial sample coordinates
        signal_vals (array): [x, signal(x)] where x are domain points
        model_output_complex (torch.cfloat tensor): Psi harmonic coefficients
        kpsi_values (array): harmonic wavenumbers
        F (float): focusing parameter (aperture size)
        dx (float): sample spacing
        xi (float): interpolation parameter for Psi argument

    Output:
        torch.cfloat tensor of reconstructed image over x_range
    """
    device = model_output_complex.device

    # Convert domain and signal to torch tensors
    domain = torch.tensor(x_range, dtype=torch.float64, device=device)
    real_signal = torch.tensor(signal_vals[0], dtype=torch.float64, device=device)   # x values
    complex_signal = torch.tensor(signal_vals[1], dtype=torch.cfloat, device=device) # signal(x)

    # Split coefficients into cosine/sine amplitudes
    cosAmps = model_output_complex.real
    sinAmps = -model_output_complex.imag  # sign convention

    # Wavenumber basis values
    wavenums = torch.tensor(kpsi_values, dtype=torch.float64, device=device)

    # ------------------------------------------------------------
    # Local helper to compute Psi phase term exp(i * Psi(sarr))
    # where Psi(sarr) = sum_k [cosAmps_k cos(k*sarr) + sinAmps_k sin(k*sarr)]
    # ------------------------------------------------------------
    def calc_psi(sarr):
        wavenum_sarr = torch.outer(sarr, wavenums)  # (len(sarr), nHarm)
        cosAmp_mat = cosAmps.unsqueeze(0)  # shape (1, nHarm)
        sinAmp_mat = sinAmps.unsqueeze(0)  # shape (1, nHarm)
        cos_terms = torch.cos(wavenum_sarr) * cosAmp_mat
        sin_terms = torch.sin(wavenum_sarr) * sinAmp_mat
        return torch.sum(cos_terms + sin_terms, dim=1)

    # ============================================================
    # Main reconstruction integral: iterate over output positions y
    # ============================================================
    image_vals = []
    for y in domain:
        y = y.item()

        # Integration window: [y - F/2, y + F/2]
        x0 = torch.max(real_signal[0], torch.tensor(y - F / 2, dtype=torch.float64, device=device))
        x1 = torch.min(real_signal[-1], torch.tensor(y + F / 2, dtype=torch.float64, device=device))

        # Restrict integration to overlapping portion of signal
        mask = (real_signal >= x0) & (real_signal <= x1)
        base = real_signal[mask]
        signal_segment = complex_signal[mask]

        if base.numel() == 0:
            # If no overlap, append zero
            image_vals.append(torch.tensor(0.0, dtype=torch.cfloat, device=device))
            continue

        # Fresnel diffraction kernel (quadratic phase term)
        waveform = torch.exp(-1j * torch.pi * (base - y) ** 2 / F)

        # Psi correction term (phase screen modulation)
        sarr = xi * base + (1 - xi) * y
        psi_vals = torch.exp(1j * calc_psi(sarr))

        # Integrand = kernel * signal * Psi term
        integrand = waveform * signal_segment * psi_vals

        # Numerical integration using trapezoidal rule
        integral = torch.trapz(integrand, base) / F

        image_vals.append(integral)

    return torch.stack(image_vals)

# ================================================================
# LOAD SETUP AND DATA
# ================================================================

# Load x-range values (first column only)
x_range_np = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values

# Load experiment setup (JSON dict)
with open(SETUP_PATH) as f:
    setup = json.load(f)
F, xi, DX = setup["F"], setup["xi"], 0.25  # Force DX=0.25
final_trim = int(F // 2)                   # Trim length (half aperture size)

# Load harmonic wavenumbers (kPsi)
kpsi_values_np = pd.read_csv(KPSI_PATH, header=None).values.flatten()

# Load signals (uscStruct values) and Psi coefficients from CSV
# Note: stored as strings, so must convert to complex carefully
signal_df = pd.read_csv(SIGNAL_PATH, dtype=str).map(convert_to_complex).T
psi_df = pd.read_csv(PSI_PATH, dtype=str).map(lambda s: complex(s.replace("i", "j"))).T
assert signal_df.shape[0] == psi_df.shape[0], "Mismatch between signals and Psi coefficients"

# ================================================================
# MAIN PROCESSING LOOP
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Storage for all reconstructed images
image_focused_all = []
image_unfocused_all = []

for i in tqdm(range(signal_df.shape[0]), desc="Processing samples"):
    # Extract ith sample signal and Psi coefficients
    signal_full = np.array(signal_df.iloc[i].values)
    psi_coeffs = psi_df.iloc[i].values

    # Skip empty/NaN-only signals
    if np.all(np.isnan(signal_full)):
        print(f"Skipping sample {i} (all NaNs)")
        continue

    # Replace NaNs with zeros (preserve length but avoid integration errors)
    signal_cleaned = np.nan_to_num(signal_full, nan=0.0)

    # Ensure signal length matches domain
    if signal_cleaned.shape != x_range_np.shape:
        print(f"Skipping sample {i} due to shape mismatch")
        continue

    # Package [x, signal(x)] for integral
    signal_vals = np.stack([x_range_np, signal_cleaned])

    # Convert Psi coefficients to torch tensor
    model_output_complex = torch.tensor(psi_coeffs, dtype=torch.cfloat, device=device)

    # Psi coefficients must align with wavenumbers
    if len(psi_coeffs) != len(kpsi_values_np):
        print(f"Skipping sample {i} due to length mismatch in Psi coefficients")
        continue

    try:
        # Focused reconstruction (using Psi coefficients)
        image_focused = compute_image_integral_torch(x_range_np, signal_vals, model_output_complex, kpsi_values_np, F, DX, xi)

        # Unfocused reconstruction (Psi coefficients set to zero)
        image_unfocused = compute_image_integral_torch(x_range_np, signal_vals, torch.zeros_like(model_output_complex), kpsi_values_np, F, DX, xi)

        # Convert back to numpy for storage
        image_focused = image_focused.detach().cpu().numpy()
        image_unfocused = image_unfocused.detach().cpu().numpy()

        # Ensure enough length remains after trimming
        if len(image_focused) <= 2 * final_trim:
            print(f"Skipping sample {i} due to insufficient length after trim")
            continue

        # Symmetric trimming (remove edges corresponding to aperture half-width)
        image_focused_trimmed = image_focused[final_trim:-final_trim]
        image_unfocused_trimmed = image_unfocused[final_trim:-final_trim]

        # Accumulate results
        image_focused_all.append(image_focused_trimmed)
        image_unfocused_all.append(image_unfocused_trimmed)

    except Exception as e:
        print(f"Skipping sample {i} due to exception: {e}")
        continue

# ================================================================
# FINAL SAVE
# ================================================================

# Trimmed x-range (same length as reconstructed images)
x_trimmed = x_range_np[final_trim:-final_trim]

# Save datasets to CSV
pd.DataFrame(image_focused_all).to_csv(OUTPUT_PATH_FOCUSED, index=False)
pd.DataFrame(image_unfocused_all).to_csv(OUTPUT_PATH_UNFOCUSED, index=False)
pd.DataFrame(x_trimmed).to_csv(X_TRIM_PATH, index=False, header=False)

print(f"Saved focused image dataset to: {OUTPUT_PATH_FOCUSED}")
print(f"Saved unfocused image dataset to: {OUTPUT_PATH_UNFOCUSED}")
print(f"Saved trimmed x-range to: {X_TRIM_PATH}")
