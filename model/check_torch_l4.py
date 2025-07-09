import numpy as np
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import torch

# === Load Config and Metadata ===
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
setup = json.load(open(config['paths']['setup_file_path']))
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()
DX = 0.25
SAMPLE_IDX = 0

# === Helpers ===
def convert_to_complex(s):
    return 0 if s == "NaNNaNi" else complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(vector):
    amp = np.abs(vector)
    amp_max = np.max(amp)
    return vector if amp_max == 0 else vector / amp_max

# === Load Signal ===
signal_df = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str).T
signal_row = signal_df.iloc[SAMPLE_IDX].apply(convert_to_complex).values.astype(np.complex64)
signal_row = normalize_complex_to_unit_range(signal_row)

# === Trim Zeros ===
nonzero_mask = np.abs(signal_row) > 1e-12
nonzero_indices = np.where(nonzero_mask)[0]
start_idx, end_idx = nonzero_indices[0], nonzero_indices[-1]
x_range_trunc = x_range[start_idx:end_idx+1]
signal_trunc = signal_row[start_idx:end_idx+1]

# === Shared Psi Coeffs (All Zeros) ===
num_coeffs = len(kpsi_values)
cos_coeffs_np = np.zeros(num_coeffs)
sin_coeffs_np = np.zeros(num_coeffs)
cos_coeffs_torch = torch.zeros(num_coeffs, dtype=torch.float32)
sin_coeffs_torch = torch.zeros(num_coeffs, dtype=torch.float32)
kpsi_torch = torch.tensor(kpsi_values, dtype=torch.float32)
x_torch = torch.tensor(x_range_trunc, dtype=torch.float32)
signal_torch = torch.tensor(signal_trunc, dtype=torch.complex64)

# === NumPy Implementation ===
def eval_image_numpy(x_vals, signal, kpsi, cos_coeffs, sin_coeffs, F, dx, xi):
    result = np.zeros_like(x_vals, dtype=np.complex128)
    for i, y in enumerate(x_vals):
        x0, x1 = max(x_vals[0], y - F/2), min(x_vals[-1], y + F/2)
        mask = (x_vals >= x0) & (x_vals <= x1)
        base = x_vals[mask]
        signal_vals = signal[mask]
        sarr = xi * base + (1 - xi) * y
        outer = np.outer(sarr, kpsi)
        psi_phase = cos_coeffs @ np.cos(outer.T) - sin_coeffs @ np.sin(outer.T)
        psi_vals = np.exp(1j * psi_phase)
        waveform = np.exp(-1j * np.pi * (base - y) ** 2 / F)
        integrand = waveform * signal_vals * psi_vals
        result[i] = trapezoid(integrand, base, dx=dx) / F
    return result

# === Torch Implementation ===
def eval_image_torch(x_vals, signal, kpsi, cos_coeffs, sin_coeffs, F, dx, xi):
    result = []
    for y in x_vals:
        x0, x1 = max(x_vals[0], y - F/2), min(x_vals[-1], y + F/2)
        mask = (x_vals >= x0) & (x_vals <= x1)
        base = x_vals[mask]
        signal_vals = signal[mask]
        sarr = xi * base + (1 - xi) * y
        outer = torch.outer(sarr, kpsi)
        psi_phase = torch.sum(
            cos_coeffs.unsqueeze(1) * torch.cos(outer.T) -
            sin_coeffs.unsqueeze(1) * torch.sin(outer.T),
            dim=0)
        psi_vals = torch.exp(1j * psi_phase)
        waveform = torch.exp(-1j * np.pi * (base - y) ** 2 / F)
        integrand = waveform * signal_vals * psi_vals
        integral = torch.trapz(integrand.real, dx=dx) + 1j * torch.trapz(integrand.imag, dx=dx)
        result.append(integral / F)
    return torch.stack(result)

# === Evaluate Image Integrals ===
image_np = eval_image_numpy(x_range_trunc, signal_trunc, kpsi_values, cos_coeffs_np, sin_coeffs_np, F, DX, xi)
image_torch = eval_image_torch(x_torch, signal_torch, kpsi_torch, cos_coeffs_torch, sin_coeffs_torch, F, DX, xi)

# === L4 Loss ===
l4_np = -np.sum(np.abs(image_np)**4) * DX
l4_torch = -torch.sum(torch.abs(image_torch)**4).item() * DX

# === Print Results ===
print(f"L4 Loss (NumPy):  {l4_np:.6f}")
print(f"L4 Loss (Torch):  {l4_torch:.6f}")
print(f"Abs Difference:   {abs(l4_np - l4_torch):.6e}")

# === Optional Plot ===
plt.figure(figsize=(10, 4))
plt.plot(x_range_trunc, np.abs(image_np), label="NumPy")
plt.plot(x_range_trunc, torch.abs(image_torch).cpu().numpy(), '--', label="Torch")
plt.title("Unfocused Image Amplitude")
plt.xlabel("x")
plt.ylabel("|I(x)|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_unfocused_integral.png")
