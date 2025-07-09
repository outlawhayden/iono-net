# === Improved PyTorch L4 Training Script with Lazy Loading ===

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import yaml
import json
import pickle
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# === Load Config ===
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("config_used.csv", "w") as f:
    for key, val in config.items():
        f.write(f"{key},{val}\n")

with open("training_losses_torch.csv", "w") as f:
    f.write("Epoch,TrainLoss,TestLoss\n")

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Helper Functions ===
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def normalize_complex_to_unit_range(vector):
    amp = np.abs(vector)
    amp_max = np.max(amp)
    if amp_max == 0:
        return vector
    return vector / amp_max

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

def compute_l4_image_loss_torch_no_class(
    x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5
):
    """
    Compute differentiable L4 image-domain loss directly from signal and predicted Fourier coefficients.

    Parameters:
        x_range:                (N,) array of spatial positions (numpy array)
        signal_vals:           (2, N) array with [x_vals; complex_signal] (numpy array)
        model_output_complex:  (Nh,) numpy array of complex-valued predicted Fourier coefficients
        kpsi_values:           (Nh,) array of wavenumbers (numpy array)
        F:                     scalar focusing aperture (float)
        dx:                    scalar integration step (float)
        xi:                    scalar mix parameter for s(x,y)

    Returns:
        l4_loss: scalar torch tensor (differentiable)
    """
    # === Convert inputs to torch tensors ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain = torch.tensor(x_range, dtype=torch.float64, device=device)
    real_signal = torch.tensor(signal_vals[0], dtype=torch.float64, device=device)
    complex_signal = torch.tensor(signal_vals[1], dtype=torch.cfloat, device=device)
    signal_tensor = torch.stack([real_signal, complex_signal])
    cosAmps = model_output_complex.real
    sinAmps = -model_output_complex.imag

    wavenums = torch.tensor(kpsi_values, dtype=torch.float64, device=device)
    F = torch.tensor(F, dtype=torch.float64, device=device)
    dx = torch.tensor(dx, dtype=torch.float64, device=device)

    # === Inner helper: compute psi(y) via Fourier series ===
    def calc_psi(sarr):
        wavenum_sarr = torch.outer(sarr, wavenums)  # [Ns, Nh]

        # Reshape cosAmps/sinAmps to [1, Nh] for broadcasting to [Ns, Nh]
        cosAmp_mat = cosAmps.unsqueeze(0)  # [1, Nh]
        sinAmp_mat = sinAmps.unsqueeze(0)  # [1, Nh]

        cos_terms = torch.cos(wavenum_sarr) * cosAmp_mat  # [Ns, Nh]
        sin_terms = torch.sin(wavenum_sarr) * sinAmp_mat

        return torch.sum(cos_terms + sin_terms, dim=1)  # [Ns]

    # === Image integral ===
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

    image_vec = torch.stack(image_vals)  # (Ny,) complex
    l4_loss = -torch.sum(torch.abs(image_vec) ** 4) * dx
    return l4_loss






# === Lazy Dataset Class ===
class LazySignalDataset(Dataset):
    def __init__(self, signal_file, label_file, max_samples=None):
        self.signal_file = signal_file
        self.label_file = label_file
        self.signal_df = pd.read_csv(signal_file, dtype=str)
        self.label_df = pd.read_csv(label_file, dtype=str)

        # Transpose to get samples as rows if they were originally columns
        self.signal_df = self.signal_df.T.reset_index(drop=True)
        self.label_df = self.label_df.T.reset_index(drop=True)

        self.len = min(len(self.signal_df), len(self.label_df))
        if max_samples:
            self.len = min(self.len, max_samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        signal_row = self.signal_df.iloc[idx].apply(convert_to_complex).values.astype(np.complex64)
        label_row = self.label_df.iloc[idx].apply(convert_to_complex).values.astype(np.complex64)

        signal_row = normalize_complex_to_unit_range(signal_row)
        label_row = normalize_complex_to_unit_range(label_row)

        signal = split_complex_to_imaginary(signal_row)
        label = split_complex_to_imaginary(label_row)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def calculate_l4_batch_subsample(batch_x, preds_real, preds_imag, x_range_tensor, kpsi_tensor, F, DX, xi, zero_pad, num_samples=2):
    indices = torch.randperm(batch_x.shape[0])[:num_samples]
    l4 = 0.0

    for i in indices:
        # Construct complex signal from real+imag
        signal_real = x_range_tensor  # already (N,)
        signal_imag = batch_x[i][1441:]
        signal_re = batch_x[i][:1441]
        signal_cplx = signal_re.to(torch.float64) + 1j * signal_imag.to(torch.float64)

        # Stack [x_vals; signal_cplx] as [2, N]
        signal_vals = torch.stack([signal_real.to(torch.float64), signal_cplx])

        # Merge predicted output into complex vector
        model_output_complex = preds_real[i].to(torch.float64) + 1j * preds_imag[i].to(torch.float64)

        l4 += compute_l4_image_loss_torch_no_class(
            x_range_tensor.cpu().numpy(),
            signal_vals,
            model_output_complex,
            kpsi_tensor.cpu().numpy(),
            F, DX, xi
        )

    return l4 / num_samples


# === Model Definition ===
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

# === Load Meta Data ===
x_range = pd.read_csv(config['paths']['x_range_file_path']).iloc[:, 0].values
x_range_tensor = torch.tensor(x_range, dtype=torch.float32).to(device)
setup = json.load(open(config['paths']['setup_file_path']))
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]
kpsi_values = pd.read_csv(config['paths']['kpsi_file_path']).values.squeeze()
kpsi_tensor = torch.tensor(kpsi_values, dtype=torch.float32).to(device)
zero_pad = 50
num_l4_samples = 4

# === Dataset and Loader ===
train_dataset = LazySignalDataset(
    config['paths']['signal_data_file_path'],
    config['paths']['label_data_file_path'],
    max_samples=config['training'].get('max_train_samples', None)
)
test_dataset = LazySignalDataset(
    config['paths']['test_data_file_path'],
    config['paths']['test_label_file_path'],
    max_samples=config['training'].get('max_test_samples', None)
)
data_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# === Initialize Model & Optimizer ===
input_dim = train_dataset[0][0].shape[0]
activation_cls = getattr(nn, config['model']['activation'])
dropout_rate = config['model'].get('dropout_rate', 0.4)
model = ConfigurableModel(config['model']['architecture'], activation_fn=activation_cls, input_dim=input_dim, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate']['fixed'], weight_decay=config['training'].get('l2_reg_weight', 0.0))

# === Loss Weights ===
l4_weight = config['training'].get('l4_weight', 0)
fourier_weight = config['training'].get('fourier_weight', 0)
fourier_d1_weight = config['training'].get('fourier_d1_weight', 0)
fourier_d2_weight = config['training'].get('fourier_d2_weight', 0)

# === Training Loop ===
loss_history = []
for epoch in tqdm(range(config['optimizer']['maxiter_adam'])):
    model.train()
    total_loss = 0.0
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)):
        optimizer.zero_grad()
        preds = model(batch_x.to(device))
        preds_real, preds_imag = preds[:, :6], preds[:, 6:]
        true_real, true_imag = batch_y[:, :6].to(device), batch_y[:, 6:].to(device)

        sq_diffs = (preds_real - true_real) ** 2 + (preds_imag - true_imag) ** 2
        direct_loss = torch.mean(torch.sum(sq_diffs, dim=1))
        idx = torch.arange(6, device=device).float()
        d1_loss = torch.mean(torch.sum((idx ** 2) * sq_diffs, dim=1))
        d2_loss = torch.mean(torch.sum((idx ** 4) * sq_diffs, dim=1))

        l4_loss = torch.tensor(0.0, device=device)
        if l4_weight > 0:
            start = time.time() # make sure L4 loss term is negative
            l4_loss = -1 * calculate_l4_batch_subsample(
                batch_x.to(device), preds_real, preds_imag,
                x_range_tensor, kpsi_tensor, F, 0.25, xi, zero_pad,
                num_samples=num_l4_samples
            )
        total = (fourier_weight * direct_loss +
                 fourier_d1_weight * d1_loss +
                 fourier_d2_weight * d2_loss +
                 l4_weight * l4_loss)

        total.backward()
        optimizer.step()
        total_loss += total.item()

    test_epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for test_batch_x, test_batch_y in test_loader:
            test_preds = model(test_batch_x.to(device))
            test_real, test_imag = test_preds[:, :6], test_preds[:, 6:]
            test_true_real = test_batch_y[:, :6].to(device)
            test_true_imag = test_batch_y[:, 6:].to(device)
            test_sq_diff = (test_real - test_true_real) ** 2 + (test_imag - test_true_imag) ** 2
            test_loss = torch.mean(torch.sum(test_sq_diff, dim=1)) * fourier_weight
            test_epoch_loss += test_loss.item()
    model.train()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")
    loss_history.append((epoch + 1, total_loss, test_epoch_loss))
    with open("training_losses_torch.csv", "a") as f:
        f.write(f"{epoch},{total_loss},{test_epoch_loss}\n")

# === Save Weights ===
with open("model_weights_torch.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

print("Training complete.")
