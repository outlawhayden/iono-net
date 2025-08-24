# === Imports and Setup ===
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import textwrap

# === Torch device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Matplotlib Settings ===
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (40, 10)
plt.rcParams["savefig.dpi"] = 300

# === Parameters ===
SAMPLE_IDX = 0
DX = 0.25
zero_pad = 50
ZOOM_RADIUS = 50
ISLR_RADIUS = 5
ISLR_RADIUS_RATIO = 0.6
ISLR_MAIN_LOBE_WIDTH = 0.75
FILENAME_PREFIX = "torch_eval"
DATA_DIR = "/home/houtlaw/iono-net/data/aug25"
MODEL_WEIGHTS_PATH = "/home/houtlaw/iono-net/model/torch_model_weights_multi.pkl"
PLOT_DIR = "visualization_outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

# === File Paths ===
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250707_155325.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250707_155325.json"
SCATTERER_PATH = f"{DATA_DIR}/test_nuStruct_withSpeckle_20250707_155322.csv"
SIGNAL_PATH = f"{DATA_DIR}/test_uscStruct_vals_20250707_155324.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250707_155325.csv"
PSI_COEFFS_PATH = f"{DATA_DIR}/test_compl_ampls_20250707_155324.csv"

# === Plotting Constants ===
LINEWIDTHS = {"true": 4, "nn": 3, "unfocused": 3}
COLORS = {"true": "black", "nn": "red", "unfocused": "gray"}
FONTSIZE_TITLE = 30
FONTSIZE_LABEL = 22
FONTSIZE_LEGEND = 20
FONTSIZE_SETUP = 14

# === Helper Functions ===
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    return matrix / amp_max

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

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

def compute_islr(image_integral, known_scatterers, x_vals, radius, radius_ratio, main_lobe_width, dx):
    islrs = []
    peaks = [x_vals[i] for i, scatterer in enumerate(known_scatterers) if scatterer > 2]
    image_integral = image_integral ** 2
    for peak in peaks:
        inner_indices = [i for i, x in enumerate(x_vals) if abs(x - peak) <= main_lobe_width]
        outer_indices = [i for i, x in enumerate(x_vals) if main_lobe_width < abs(x - peak) <= (radius * radius_ratio)]
        if not inner_indices or not outer_indices:
            continue
        inner_integral = np.trapz(image_integral[inner_indices], x_vals[inner_indices])
        outer_integral = np.trapz(image_integral[outer_indices], x_vals[outer_indices])
        islr = 10 * np.log10(outer_integral / inner_integral) if inner_integral != 0 else 0
        islrs.append(islr)
    return np.mean(islrs)

# === Load Data ===
x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values
setup = json.load(open(SETUP_PATH))
F, xi = setup["F"], setup["xi"]
kpsi_values = pd.read_csv(KPSI_PATH).values.flatten()
setup_str = ", ".join(f"{k}={v}" for k, v in setup.items())
wrapped_setup = textwrap.fill(setup_str, width=180)

scatterer_mag = pd.read_csv(SCATTERER_PATH).map(convert_to_complex).iloc[:, SAMPLE_IDX].map(np.abs).values
signal_data = pd.read_csv(SIGNAL_PATH).map(convert_to_complex).T.iloc[SAMPLE_IDX].values
psi_coeffs_vals = pd.read_csv(PSI_COEFFS_PATH).T.map(lambda x: complex(x.replace('i', 'j'))).iloc[SAMPLE_IDX].values

signal_data = normalize_complex_to_unit_range(signal_data[None, :])[0]

# === Model ===
class ConfigurableModel(nn.Module):
    def __init__(self, architecture, activation_fn=torch.relu, dropout_rate=0.0, input_dim=2882, output_dim=12):
        super().__init__()
        layers = []
        in_features = input_dim
        for size in architecture:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation_fn())
            in_features = size
        layers.append(nn.Linear(in_features, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

# === Prepare Signal and Model ===
signal_trimmed = signal_data[4 * zero_pad : -4 * zero_pad]
x_range_trunc = x_range[4 * zero_pad : -4 * zero_pad]
signal_vals_trunc = np.vstack((x_range_trunc, signal_trimmed))
signal_input = split_complex_to_imaginary(signal_data).astype(np.float32)[None, :]
signal_tensor = torch.tensor(signal_input, dtype=torch.float32).to(device)
input_dim = signal_tensor.shape[1]

model = ConfigurableModel(architecture=[64,64,64,64], activation_fn=nn.ReLU, input_dim=input_dim).to(device)
with open(MODEL_WEIGHTS_PATH, 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

# === Inference ===
with torch.no_grad():
    model_output = model(signal_tensor)[0]
half = model_output.shape[0] // 2
model_output_complex = model_output[:half] + 1j * model_output[half:]

# === Convert Coeffs and Compute Images ===
psi_true_torch = torch.tensor(psi_coeffs_vals, dtype=torch.cfloat, device=device)
psi_pred_torch = model_output_complex.to(torch.cfloat)
image_true = compute_image_integral_torch(x_range_trunc, signal_vals_trunc, psi_true_torch, kpsi_values, F, DX, xi)
image_pred = compute_image_integral_torch(x_range_trunc, signal_vals_trunc, psi_pred_torch, kpsi_values, F, DX, xi)
image_unfocused = compute_image_integral_torch(x_range_trunc, signal_vals_trunc, torch.zeros_like(psi_pred_torch), kpsi_values, F, DX, xi)

# === Plotting ===
def plot_curve(x, curves, labels, colors, title, filename):
    fig, ax = plt.subplots()
    for y, label, color in zip(curves, labels, colors):
        ax.plot(x, y, label=label, lw=3, color=color)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.set_xlabel("x", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.02, wrapped_setup, ha='center', fontsize=FONTSIZE_SETUP)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(PLOT_DIR, filename))

plot_curve(
    x_range_trunc,
    [np.abs(image_true.detach().cpu()) / DX, np.abs(image_pred.detach().cpu()) / DX],
    ["True Ψ", "NN Ψ"],
    [COLORS["true"], COLORS["nn"]],
    "Image Integral Comparison (PyTorch)",
    "image_integral_comparison.png"
)

plot_curve(
    x_range_trunc,
    [np.abs(image_unfocused.detach().cpu()) / DX, np.abs(image_pred.detach().cpu()) / DX],
    ["Unfocused (Ψ=0)", "NN Ψ"],
    [COLORS["unfocused"], COLORS["nn"]],
    "Unfocused vs Focused Image (NN Ψ)",
    "unfocused_vs_nn.png"
)

# === Zoomed-In Scatterers ===
scatterer_indices = np.where(scatterer_mag > 1)[0]
trim_offset = 4 * zero_pad
adjusted_indices = [i - trim_offset for i in scatterer_indices if trim_offset <= i < trim_offset + len(x_range_trunc)]

for i, idx in enumerate(adjusted_indices):
    start, end = max(0, idx - ZOOM_RADIUS), min(len(x_range_trunc), idx + ZOOM_RADIUS)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_range_trunc[start:end], np.abs(image_true[start:end].cpu()) / DX, COLORS["true"], label="True Focus", lw=LINEWIDTHS["true"])
    ax.plot(x_range_trunc[start:end], np.abs(image_pred[start:end].cpu()) / DX, COLORS["nn"], label="NN Focus", lw=LINEWIDTHS["nn"])
    ax.axvline(x_range_trunc[idx], color='k', linestyle='--', label='Scatterer', lw=2)
    ax.set_title(f"Zoomed-In Scatterer at x ≈ {x_range_trunc[idx]:.2f}", fontsize=FONTSIZE_TITLE - 4)
    ax.set_xlabel("x", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
    ax.grid(True)
    ax.legend(fontsize=FONTSIZE_LEGEND - 2)
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.02, wrapped_setup, ha='center', fontsize=FONTSIZE_SETUP)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(PLOT_DIR, f"zoomed_scatterer_peak_{i}.png"))
    plt.close()

# === Psi Plot ===
def build_psi_values(kpsi, compl_ampls, x):
    return np.sum([np.real(c * np.exp(1j * k * x)) for k, c in zip(kpsi, compl_ampls)], axis=0)

psi_pred_vals = build_psi_values(kpsi_values, model_output_complex.detach().cpu().numpy(), x_range_trunc)
psi_true_vals = build_psi_values(kpsi_values, psi_coeffs_vals, x_range_trunc)

plot_curve(
    x_range_trunc,
    [psi_true_vals, psi_pred_vals],
    ["True Psi", "NN Psi"],
    [COLORS["true"], COLORS["nn"]],
    "Psi Function Comparison",
    "psi_comparison.png"
)

# === Metrics ===
def compute_l4(image): return -torch.sum(torch.abs(image) ** 4).item() * DX

print(f"L4 Loss (True Ψ):      {compute_l4(image_true):.6f}")
print(f"L4 Loss (NN Ψ):        {compute_l4(image_pred):.6f}")
print(f"L4 Loss (Unfocused Ψ): {compute_l4(image_unfocused):.6f}")

islr_true = compute_islr(np.abs(image_true.cpu().numpy()) / DX, scatterer_mag, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
islr_pred = compute_islr(np.abs(image_pred.cpu().numpy()) / DX, scatterer_mag, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)
islr_unfocused = compute_islr(np.abs(image_unfocused.cpu().numpy()) / DX, scatterer_mag, x_range_trunc, ISLR_RADIUS, ISLR_RADIUS_RATIO, ISLR_MAIN_LOBE_WIDTH, DX)

print(f"ISLR (True Ψ):      {islr_true:.3f} dB")
print(f"ISLR (NN Ψ):        {islr_pred:.3f} dB")
print(f"ISLR (Unfocused Ψ): {islr_unfocused:.3f} dB")
