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
import yaml

# === Torch device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("/home/houtlaw/iono-net/model/config_ynet.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Matplotlib Settings ===
plt.rcParams.update({'font.size': 22})
rcParams["figure.figsize"] = (40, 10)
plt.rcParams["savefig.dpi"] = 300

# === Parameters ===
SAMPLE_IDX = 13
DX = 0.25
zero_pad = 50
ZOOM_RADIUS = 50
ISLR_RADIUS = 5
ISLR_RADIUS_RATIO = 0.6
ISLR_MAIN_LOBE_WIDTH = 0.75
FILENAME_PREFIX = "torch_eval"
DATA_DIR = "/home/houtlaw/iono-net/data/aug25"
MODEL_WEIGHTS_PATH = "/home/houtlaw/iono-net/model/ynet_focus_weights.pkl"
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
kpsi_values = pd.read_csv(KPSI_PATH, header=None).values.flatten()
setup_str = ", ".join(f"{k}={v}" for k, v in setup.items())
wrapped_setup = textwrap.fill(setup_str, width=180)

scatterer_mag = pd.read_csv(SCATTERER_PATH).map(convert_to_complex).iloc[:, SAMPLE_IDX].map(np.abs).values
signal_data = pd.read_csv(SIGNAL_PATH).map(convert_to_complex).T.iloc[SAMPLE_IDX].values
psi_coeffs_vals = pd.read_csv(PSI_COEFFS_PATH).T.map(lambda x: complex(x.replace('i', 'j'))).iloc[SAMPLE_IDX].values

signal_data = normalize_complex_to_unit_range(signal_data[None, :])[0]


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__(nn.Conv1d(c_in, c_out, k, padding=p, bias=False),
                         nn.BatchNorm1d(c_out),
                         nn.ReLU(inplace=True))

class YNet1D(nn.Module):
    def __init__(self, base_ch=32, depth=4, in_ch=2, out_ch=2, n_coeff=12, dropout=0.0):
        super().__init__()
        self.depth = depth
        enc, pools, ch = [], [], in_ch
        for d in range(depth):
            enc.append(ConvBNReLU(ch, base_ch * 2**d))
            pools.append(nn.MaxPool1d(2))
            ch = base_ch * 2**d
        self.enc = nn.ModuleList(enc)
        self.pools = nn.ModuleList(pools)

        self.bottle = ConvBNReLU(ch, ch * 2)
        bott_ch = ch * 2
        ch *= 2

        dec, ups = [], []
        for d in reversed(range(depth)):
            ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec.append(ConvBNReLU(ch + base_ch*2**d, base_ch*2**d))
            ch = base_ch*2**d
        self.ups = nn.ModuleList(ups)
        self.dec = nn.ModuleList(dec)
        self.head_img = nn.Conv1d(ch, out_ch, 1)
        self.drop = nn.Dropout(dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.LayerNorm(bott_ch),
                                nn.ReLU(inplace=True),
                                nn.Linear(bott_ch, n_coeff))

    def forward(self, x_flat):
        B, F = x_flat.shape
        L = F // 2
        x = x_flat.view(B, 2, L)
        skips = []
        for enc, pool in zip(self.enc, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottle(x)
        coeff_out = self.fc(self.gap(x))
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                diff = x.shape[-1] - skip.shape[-1]
                x = x[..., :-diff] if diff > 0 else nn.functional.pad(x, (0, -diff))
            x = dec(torch.cat([x, skip], dim=1))
            x = self.drop(x)

        residual = self.head_img(x).view(B, -1)

        # Match length of residual and x_flat
        residual_len = residual.shape[1]
        input_len = x_flat.shape[1]

        if residual_len < input_len:
            residual = nn.functional.pad(residual, (0, input_len - residual_len))
        elif residual_len > input_len:
            residual = residual[:, :input_len]

        img_out = x_flat + residual
        return img_out, coeff_out

        # Project to 2-channel residual
        residual = self.head_img(x).view(B, -1)

        # === Ensure residual length matches input length ===
        residual_len = residual.shape[1]
        input_len = x_flat.shape[1]
        if residual_len < input_len:
            residual = nn.functional.pad(residual, (0, input_len - residual_len))
        elif residual_len > input_len:
            residual = residual[:, :input_len]

        img_out = x_flat + residual
        return img_out, coeff_out

# === Prepare Signal and Model ===
print("sizes:", signal_data.shape, x_range.shape)
signal_trimmed = signal_data[4 * zero_pad : -4 * zero_pad]
x_range_trunc = x_range[4 * zero_pad : -4 * zero_pad]
signal_vals_trunc = np.vstack((x_range_trunc, signal_trimmed))
signal_input = split_complex_to_imaginary(signal_data).astype(np.float32)[None, :]
signal_tensor = torch.tensor(signal_input, dtype=torch.float32).to(device)
input_dim = signal_tensor.shape[1]

depth = 4
model = YNet1D(
    base_ch=12,  # Match what the weights used
    depth=9,
    in_ch=2,
    out_ch=2,
    n_coeff=12,
    dropout=0.0,
).to(device)
with open(MODEL_WEIGHTS_PATH, 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

print("model loaded...")

def negative_l4_norm(pred_flat):
    B, F = pred_flat.shape
    L = F // 2
    real, imag = pred_flat[:, :L], pred_flat[:, L:]
    amp4 = (real**2 + imag**2)**2
    return -torch.mean(torch.sum(amp4, dim=1))

lambda_l4    = cfg['loss']['l4_weight']
lambda_coef  = cfg['loss']['coef_weight']
lambda_recon = cfg['loss']['recon_accuracy_weight']

def forward_with_padding(xb):
    m = 2 ** depth; pad = (-xb.shape[1]) % m
    if pad: xb = nn.functional.pad(xb, (0, pad))
    img, coef = model(xb)
    if pad: img = img[:, :-pad]
    return img, coef

def focus_pytorch(x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5):
    _device = signal_vals.device
    domain = x_range.to(dtype=torch.float64, device=_device)
    wavenums = kpsi_values.to(dtype=torch.float64, device=_device)
    F = F.to(dtype=torch.float64, device=_device)
    dx = dx.to(dtype=torch.float64, device=_device)
    real_signal = signal_vals[0].real
    complex_signal = signal_vals[1]
    cosAmps = model_output_complex.real
    sinAmps = model_output_complex.imag

    def calc_psi(sarr):
        sarr = sarr.flatten()
        wnum = torch.outer(sarr, wavenums)
        cos_terms = torch.cos(wnum) * cosAmps
        sin_terms = torch.sin(wnum) * sinAmps
        return torch.sum(cos_terms - sin_terms, dim=1)

    image_vals = []
    for y in domain:
        x0 = torch.max(real_signal[0], y - F / 2)
        x1 = torch.min(real_signal[-1], y + F / 2)
        mask = (real_signal >= x0) & (real_signal <= x1)
        if not mask.any():
            image_vals.append(torch.tensor(0.0, dtype=torch.cfloat, device=_device))
            continue
        base = real_signal[mask]
        signal_segment = complex_signal[mask]
        waveform = torch.exp(-1j * torch.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y.item()
        psi_vals = torch.exp(1j * calc_psi(sarr))
        integrand = waveform * signal_segment * psi_vals
        image_vals.append(torch.trapz(integrand, base) / F)

    return torch.stack(image_vals)

def recon_accuracy_loss(x_range, psi_coeffs, image_pred, signal_vals, kpsi_vals, F, dx, xi=0.5):
    real, imag = psi_coeffs[:, :6], psi_coeffs[:, 6:]
    psi_cplx = torch.complex(real, imag)  # (B, 6)

    focus_pred = focus_pytorch(x_range, signal_vals, psi_cplx[0], kpsi_vals, F, dx, xi)
    mag_focus = torch.abs(focus_pred).float()  # (L,)

    B, Fflat = image_pred.shape
    L = Fflat // 2
    img_complex = image_pred.view(B, 2, L)
    mag_pred = torch.sqrt(img_complex[:, 0]**2 + img_complex[:, 1]**2)  # (B, L)

    # Broadcast focus image across batch
    if B > 1:
        mag_focus = mag_focus.unsqueeze(0).expand(B, -1)

    # === Fix shape mismatch ===
    if mag_pred.shape[1] != mag_focus.shape[1]:
        min_len = min(mag_pred.shape[1], mag_focus.shape[1])
        mag_pred = mag_pred[:, :min_len]
        mag_focus = mag_focus[:, :min_len]

    return nn.functional.mse_loss(mag_pred, mag_focus)



# === Inference ===
with torch.no_grad():
    image_out, coef_out = forward_with_padding(signal_tensor)

print("input_size:", signal_tensor.shape)
print("output_size:", image_out.shape)

half = coef_out.shape[1] // 2
model_output_complex = coef_out[0, :half] + 1j * coef_out[0, half:]


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

# === Direct Image Prediction Plot ===
# Extract model direct image output
img_out_real = image_out[0, : image_out.shape[1] // 2]
img_out_imag = image_out[0, image_out.shape[1] // 2 :]
img_out_cplx = img_out_real + 1j * img_out_imag
img_out_mag = np.abs(img_out_cplx.detach().cpu().numpy()) / DX

# Load true focused image (yb)
true_img_path = cfg["paths"]["focused_val"]
yb_vals = pd.read_csv(true_img_path).values[SAMPLE_IDX]
print("yb_vals.shape:", yb_vals.shape)
yb_mag = np.abs(np.array([convert_to_complex(x) for x in yb_vals]))/ DX

print("yb_mag_size", yb_mag.shape)
print("img_out_mag_shape", img_out_mag.shape)
print("x_range", x_range_trunc.shape)

#[yb_mag[3*zero_pad:-3*zero_pad], img_out_mag[4*zero_pad:-4*zero_pad]],
# === Combined Plotting: 4 Subplots Vertically ===
fig, axes = plt.subplots(4, 1, figsize=(40, 40))  # 4 rows, 1 column

# --- Subplot 1: True Known Image (yb) ---
axes[0].plot(
    x_range_trunc[zero_pad:-zero_pad],
    yb_mag[4*zero_pad:-4*zero_pad],
    lw=3,
    color=COLORS["true"],
)
axes[0].set_title("True Focused Image (yb)", fontsize=FONTSIZE_TITLE)
axes[0].set_xlabel("x", fontsize=FONTSIZE_LABEL)
axes[0].set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
axes[0].grid(True)

# --- Subplot 2: Direct NN Output ---
axes[1].plot(
    x_range_trunc[zero_pad:-zero_pad],
    img_out_mag[5*zero_pad:-5*zero_pad],
    lw=3,
    color=COLORS["nn"],
)
axes[1].set_title("Direct NN Image Output", fontsize=FONTSIZE_TITLE)
axes[1].set_xlabel("x", fontsize=FONTSIZE_LABEL)
axes[1].set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
axes[1].grid(True)

# --- Subplot 3: Focused with True Ψ ---
axes[2].plot(
    x_range_trunc,
    np.abs(image_unfocused.detach().cpu()) / DX,
    lw=3,
    color=COLORS["unfocused"],
)
axes[2].set_title("Unfocused Image (0 Ψ)", fontsize=FONTSIZE_TITLE)
axes[2].set_xlabel("x", fontsize=FONTSIZE_LABEL)
axes[2].set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
axes[2].grid(True)

# --- Subplot 4: Focused with NN Ψ ---
axes[3].plot(
    x_range_trunc,
    np.abs(image_pred.detach().cpu()) / DX,
    lw=3,
    color=COLORS["nn"],
)
axes[3].set_title("Image Focused with NN Ψ", fontsize=FONTSIZE_TITLE)
axes[3].set_xlabel("x", fontsize=FONTSIZE_LABEL)
axes[3].set_ylabel("Amplitude", fontsize=FONTSIZE_LABEL)
axes[3].grid(True)

# Add setup string at the bottom
fig.text(0.5, 0.01, wrapped_setup, ha="center", fontsize=FONTSIZE_SETUP)

# Adjust layout and save
fig.tight_layout(rect=[0, 0.03, 1, 1])
combined_path = os.path.join(PLOT_DIR, "combined_four_panel.png")
fig.savefig(combined_path)
plt.close(fig)

print(f"Saved combined 4-panel plot to {combined_path}")

print(f"Saved combined plot to {combined_path}")
 
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
