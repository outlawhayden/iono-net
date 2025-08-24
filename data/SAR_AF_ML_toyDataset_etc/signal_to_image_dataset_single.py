import os
import json
import pandas as pd
import numpy as np
import torch

# --- Configuration ---
DATA_DIR = "/home/houtlaw/iono-net/data/aug25"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250707_155325.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250707_155325.json"
SIGNAL_PATH = f"{DATA_DIR}/train_uscStruct_vals_20250707_155313.csv"
PSI_PATH = f"{DATA_DIR}/train_compl_ampls_20250707_155312.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250707_155325.csv"
OUTPUT_PATH_FOCUSED = f"{DATA_DIR}/train_image_recon_jnp_single.csv"
OUTPUT_PATH_UNFOCUSED = f"{DATA_DIR}/train_image_recon_jnp_unfocused_single.csv"
X_TRIM_PATH = f"{DATA_DIR}/x_range_image_recon_jnp_single.csv"

# --- Helpers ---
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace("i", "j"))

def compute_image_integral_torch(x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5):
    device = model_output_complex.device
    domain = torch.tensor(x_range, dtype=torch.float64, device=device)
    real_signal = torch.tensor(signal_vals[0], dtype=torch.float64, device=device)
    complex_signal = torch.tensor(signal_vals[1], dtype=torch.cfloat, device=device)
    cosAmps = model_output_complex.real
    sinAmps = -model_output_complex.imag
    wavenums = torch.tensor(kpsi_values, dtype=torch.float64, device=device)

    def calc_psi(sarr):
        wavenum_sarr = torch.outer(sarr, wavenums)
        cosAmp_mat = cosAmps.unsqueeze(0)
        sinAmp_mat = sinAmps.unsqueeze(0)
        cos_terms = torch.cos(wavenum_sarr) * cosAmp_mat
        sin_terms = torch.sin(wavenum_sarr) * sinAmp_mat
        return torch.sum(cos_terms + sin_terms, dim=1)

    image_vals = []
    for y in domain:
        y = y.item()
        x0 = torch.max(real_signal[0], torch.tensor(y - F / 2, dtype=torch.float64, device=device))
        x1 = torch.min(real_signal[-1], torch.tensor(y + F / 2, dtype=torch.float64, device=device))
        mask = (real_signal >= x0) & (real_signal <= x1)
        base = real_signal[mask]
        signal_segment = complex_signal[mask]
        if base.numel() == 0:
            image_vals.append(torch.tensor(0.0, dtype=torch.cfloat, device=device))
            continue
        waveform = torch.exp(-1j * torch.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_vals = torch.exp(1j * calc_psi(sarr))
        integrand = waveform * signal_segment * psi_vals
        integral = torch.trapz(integrand, base) / F
        image_vals.append(integral)
    return torch.stack(image_vals)


# --- Load setup and data ---
x_range_np = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values
with open(SETUP_PATH) as f:
    setup = json.load(f)
F, xi, DX = setup["F"], setup["xi"], 0.25
kpsi_values_np = pd.read_csv(KPSI_PATH, header=None).values.flatten()

signal_df = pd.read_csv(SIGNAL_PATH, dtype=str).map(convert_to_complex).T
psi_df = pd.read_csv(PSI_PATH, dtype=str).map(lambda s: complex(s.replace("i", "j"))).T

assert signal_df.shape[0] == psi_df.shape[0]

# --- Process only the first sample ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i = 0

signal_full = np.array(signal_df.iloc[i].values)
psi_coeffs = psi_df.iloc[i].values

if np.all(np.isnan(signal_full)):
    raise ValueError(f"Sample {i} is all NaNs")

signal_cleaned = np.nan_to_num(signal_full, nan=0.0)
if signal_cleaned.shape != x_range_np.shape:
    raise ValueError(f"Signal shape {signal_cleaned.shape} ≠ x_range shape {x_range_np.shape}")

signal_vals = np.stack([x_range_np, signal_cleaned])
model_output_complex = torch.tensor(psi_coeffs, dtype=torch.cfloat, device=device)
if len(psi_coeffs) != len(kpsi_values_np):
    raise ValueError(f"Psi length {len(psi_coeffs)} ≠ kpsi length {len(kpsi_values_np)}")

image_focused = compute_image_integral_torch(x_range_np, signal_vals, model_output_complex, kpsi_values_np, F, DX, xi)
image_unfocused = compute_image_integral_torch(x_range_np, signal_vals, torch.zeros_like(model_output_complex), kpsi_values_np, F, DX, xi)

image_focused = image_focused.detach().cpu().numpy()
image_unfocused = image_unfocused.detach().cpu().numpy()

# Trim
x_final = x_range_np
common_length = min(len(image_focused), len(image_unfocused))
x_final = x_final[:common_length]
final_trim = int(F // 2)

if common_length <= 2 * final_trim:
    raise ValueError(f"common_length = {common_length} too short for trim of {final_trim} each side")

image_focused_arr = image_focused[:common_length][final_trim:-final_trim]
image_unfocused_arr = image_unfocused[:common_length][final_trim:-final_trim]
x_trimmed = x_final[final_trim:-final_trim]

# Save
pd.DataFrame([image_focused_arr]).to_csv(OUTPUT_PATH_FOCUSED, index=False)
pd.DataFrame([image_unfocused_arr]).to_csv(OUTPUT_PATH_UNFOCUSED, index=False)
pd.DataFrame(x_trimmed).to_csv(X_TRIM_PATH, index=False, header=False)

print(f"Saved focused image dataset to: {OUTPUT_PATH_FOCUSED}")
print(f"Saved unfocused image dataset to: {OUTPUT_PATH_UNFOCUSED}")
print(f"Saved trimmed x-range to: {X_TRIM_PATH}")
