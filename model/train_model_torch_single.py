# === Multi-Sample L4 Minimization Script ===

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import json
import pickle
from tqdm import tqdm

# === Load Config ===
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Parameters ===
NUM_SAMP = 1  # Number of samples to use in training
DX = 0.25
zero_pad = 50
csv_path = "multi_sample_l4_loss.csv"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Helper Functions ===
def convert_to_complex(s):
    return 0 if s == "NaNNaNi" else complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(vector):
    amp = np.abs(vector)
    amp_max = np.max(amp)
    return vector if amp_max == 0 else vector / amp_max

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def compute_l4_image_loss_torch_no_class(x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = torch.tensor(x_range, dtype=torch.float64, device=device)
    real_signal = torch.tensor(signal_vals[0], dtype=torch.float64, device=device)
    complex_signal = torch.tensor(signal_vals[1], dtype=torch.cfloat, device=device)
    signal_tensor = torch.stack([real_signal, complex_signal])
    cosAmps = model_output_complex.real
    sinAmps = -model_output_complex.imag # NEGATIVE

    wavenums = torch.tensor(kpsi_values, dtype=torch.float64, device=device)
    F = torch.tensor(F, dtype=torch.float64, device=device)
    dx = torch.tensor(dx, dtype=torch.float64, device=device)

    def calc_psi(sarr):
        wavenum_sarr = torch.outer(sarr, wavenums)
        cosAmp_mat = cosAmps.unsqueeze(0)
        sinAmp_mat = sinAmps.unsqueeze(0)
        cos_terms = torch.cos(wavenum_sarr) * cosAmp_mat
        sin_terms = torch.sin(wavenum_sarr) * sinAmp_mat
        return torch.sum(cos_terms + sin_terms, dim=1)

    image_vals = []
    real_signal = signal_tensor[0, :].real
    complex_signal = signal_tensor[1, :]

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
        window = torch.ones_like(base, dtype=torch.float64)

        integrand = waveform * signal_segment * window * psi_vals
        integral = torch.trapz(integrand, base) / F
        image_vals.append(integral)

    image_vec = torch.stack(image_vals)
    l4_loss = 1000 * -torch.sum(torch.abs(image_vec) ** 4) * dx
    return l4_loss

def calculate_l4_batch_subsample(batch_x, preds_real, preds_imag, x_range_tensor, kpsi_tensor, F, DX, xi, zero_pad, num_samples=1):
    indices = torch.arange(batch_x.shape[0])[:num_samples]
    l4 = 0.0

    for i in indices:
        signal_real = x_range_tensor
        signal_imag = batch_x[i][1441:]
        signal_re = batch_x[i][:1441]
        signal_cplx = signal_re.to(torch.float64) + 1j * signal_imag.to(torch.float64)
        signal_vals = torch.stack([signal_real.to(torch.float64), signal_cplx])
        model_output_complex = preds_real[i].to(torch.float64) + 1j * preds_imag[i].to(torch.float64)

        l4 += compute_l4_image_loss_torch_no_class(
            x_range_tensor.cpu().numpy(),
            signal_vals,
            model_output_complex,
            kpsi_tensor.cpu().numpy(),
            F, DX, xi
        )

    return l4 / num_samples

# === Load Metadata ===
x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
x_range_tensor = torch.tensor(x_range, dtype=torch.float32).to(device)
setup = json.load(open(config['paths']['setup_file_path']))
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()
kpsi_tensor = torch.tensor(kpsi_values, dtype=torch.float32).to(device)

# === Load Multiple Samples ===
signal_df = pd.read_csv(config['paths']['signal_data_file_path'], dtype=str).T.reset_index(drop=True)
label_df = pd.read_csv(config['paths']['label_data_file_path'], dtype=str).T.reset_index(drop=True)

signal_samples = []
label_samples = []

for i in range(NUM_SAMP):
    signal_row = signal_df.iloc[i].apply(convert_to_complex).values.astype(np.complex64)
    label_row = label_df.iloc[i].apply(convert_to_complex).values.astype(np.complex64)
    signal_row = normalize_complex_to_unit_range(signal_row)
    label_row = normalize_complex_to_unit_range(label_row)
    signal_samples.append(split_complex_to_imaginary(signal_row))
    label_samples.append(split_complex_to_imaginary(label_row))

signal_tensor = torch.tensor(signal_samples, dtype=torch.float32).to(device)
label_tensor = torch.tensor(label_samples, dtype=torch.float32).to(device)

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

    def forward(self, x):
        return self.model(x)

input_dim = signal_tensor.shape[1]
activation_cls = getattr(nn, config['model']['activation'])
dropout_rate = config['model'].get('dropout_rate', 0.0)
model = ConfigurableModel(config['model']['architecture'], activation_fn=activation_cls, input_dim=input_dim, dropout_rate=dropout_rate).to(device)

# === Training ===
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
epochs = 1000
with open(csv_path, "w") as f:
    f.write("Epoch,L4Loss\n")

for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    output = model(signal_tensor)
    preds_real, preds_imag = output[:, :6], output[:, 6:]

    l4_loss = calculate_l4_batch_subsample(
        signal_tensor, preds_real, preds_imag,
        x_range_tensor, kpsi_tensor, F, DX, xi, zero_pad,
        num_samples=NUM_SAMP
    )

    l4_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: L4 Loss = {l4_loss.item():.6f}")
    with open(csv_path, "a") as f:
        f.write(f"{epoch+1},{l4_loss.item():.6f}\n")

# === Save Model ===
with open("torch_model_weights_multi.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)
    
print("weights saved...")
