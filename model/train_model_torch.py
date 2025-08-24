# === Improved PyTorch L4 Training Script with Lazy Loading (L4 in Test Loss) ===
# ------------------------------------------------------------------
# Purpose:
#   - Train a fully connected neural network (ConfigurableModel) to predict
#     complex Psi coefficients from radar signal data.
#   - Loss combines Fourier-domain coefficient errors and an image-domain
#     differentiable L4 loss (negative 4th norm).
#   - Supports lazy loading of large datasets from CSV.
#   - Logs both training and testing losses per epoch, including L4 term.
#   - Saves final model weights for later evaluation.
# ------------------------------------------------------------------

import os
import time
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
# Load configuration parameters from YAML (hyperparameters, paths, etc.)
with open("config_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

# Save a copy of config for reproducibility
with open("config_used.csv", "w") as f:
    for k, v in config.items():
        f.write(f"{k},{v}\n")

# Initialize training log CSV with headers
with open("training_losses_torch.csv", "w") as f:
    f.write("Epoch,TrainLoss,TestLoss\n")

# ------------------------------------------------------------------
#  Device
# ------------------------------------------------------------------
# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
#  Helper Functions
# ------------------------------------------------------------------
def convert_to_complex(s: str):
    """
    Convert CSV string entry to Python complex.
    Handles 'NaNNaNi' (MATLAB placeholder) by mapping to 0.
    """
    return 0 if s == "NaNNaNi" else complex(s.replace("i", "j"))

def normalize_complex_to_unit_range(vec: np.ndarray):
    """
    Normalize a complex vector so its maximum amplitude = 1.
    Ensures training inputs/labels are scale-invariant.
    """
    amp_max = np.abs(vec).max()
    return vec if amp_max == 0 else vec / amp_max

def split_complex_to_imaginary(arr: np.ndarray):
    """
    Split complex-valued array into concatenated real/imag parts.
    Example: [a+ib, c+id] -> [a, c, b, d]
    """
    return np.concatenate([arr.real, arr.imag], axis=-1)

def compute_l4_image_loss_torch_no_class(
    x_range, signal_vals, model_output_complex, kpsi_values, F, dx, xi=0.5
):
    """
    Compute differentiable L4 image-domain loss (negative L4 norm).
    
    Inputs:
        x_range: spatial domain points (numpy array)
        signal_vals: torch.Tensor([x, signal(x)]) with complex values
        model_output_complex: predicted Psi coefficients (torch.cfloat)
        kpsi_values: harmonic wavenumbers (numpy array)
        F: focusing aperture parameter
        dx: integration step size
        xi: interpolation weighting parameter

    Output:
        Scalar torch.Tensor (negative L4 loss).
    """
    _device = signal_vals.device

    # Convert to tensors on the right device
    domain = torch.tensor(x_range, dtype=torch.float64, device=_device)
    real_signal = signal_vals[0].real  # x positions
    complex_signal = signal_vals[1]    # signal(x)

    # Split Psi coefficients into cosine/sine amplitude contributions
    cosAmps = model_output_complex.real
    sinAmps = model_output_complex.imag
    wavenums = torch.tensor(kpsi_values, dtype=torch.float64, device=_device)
    F = torch.tensor(F, dtype=torch.float64, device=_device)
    dx = torch.tensor(dx, dtype=torch.float64, device=_device)

    def calc_psi(sarr):
        """Compute Psi(sarr) using harmonic expansion."""
        wnum = torch.outer(sarr, wavenums)          # shape [Ns, Nharm]
        cos_terms = torch.cos(wnum) * cosAmps
        sin_terms = torch.sin(wnum) * sinAmps
        return torch.sum(cos_terms - sin_terms, dim=1)

    # Loop over image-domain output positions
    image_vals = []
    for y in domain:
        # Restrict integration domain to [y-F/2, y+F/2]
        x0 = torch.max(real_signal[0], torch.tensor(y - F / 2, device=_device))
        x1 = torch.min(real_signal[-1], torch.tensor(y + F / 2, device=_device))
        mask = (real_signal >= x0) & (real_signal <= x1)

        if not mask.any():
            # Append zero if no overlap
            image_vals.append(torch.tensor(0.0, dtype=torch.cfloat, device=_device))
            continue

        base = real_signal[mask]
        signal_segment = complex_signal[mask]

        # Fresnel kernel
        waveform = torch.exp(-1j * torch.pi * (base - y) ** 2 / F)

        # Psi modulation
        sarr = xi * base + (1 - xi) * y
        psi_vals = torch.exp(1j * calc_psi(sarr))

        # Integrand
        integrand = waveform * signal_segment * psi_vals
        image_vals.append(torch.trapz(integrand, base) / F)

    image_vec = torch.stack(image_vals)
    # Return *negative* L4 norm (as per specification)
    return -torch.sum(torch.abs(image_vec) ** 4) * dx

def calculate_l4_batch_subsample(
    batch_x, preds_real, preds_imag, x_range_tensor,
    kpsi_tensor, F, DX, xi, zero_pad, num_samples=2):
    """
    Compute average L4 loss for a random subsample of batch.
    - Reduces cost by only evaluating L4 on a subset.
    """
    idxs = torch.randperm(batch_x.size(0))[:num_samples]
    l4_total = 0.0
    half = batch_x.size(1) // 2  # split real/imag channels

    for i in idxs:
        # Reconstruct complex signal from input split
        sig_re = batch_x[i, :half].to(torch.float64)
        sig_im = batch_x[i, half:].to(torch.float64)
        sig_cplx = sig_re + 1j * sig_im
        signal_vals = torch.stack([x_range_tensor.to(torch.float64), sig_cplx])

        # Combine predicted coefficients into complex
        coeff_cplx = (
            preds_real[i].to(torch.float64) + 1j * preds_imag[i].to(torch.float64)
        )

        # Compute image-domain L4 loss
        l4_total += compute_l4_image_loss_torch_no_class(
            x_range_tensor.cpu().numpy(),
            signal_vals,
            coeff_cplx,
            kpsi_tensor.cpu().numpy(),
            F, DX, xi,
        )
    return l4_total / num_samples

# ------------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------------
class LazySignalDataset(Dataset):
    """
    Lazy-loading dataset for signals and Psi coefficient labels.
    Loads from CSV on-the-fly (row by row) instead of preloading entire file.
    """
    def __init__(self, signal_file, label_file, max_samples=None):
        # Each row in signal_df = one sample (complex as strings)
        self.signal_df = pd.read_csv(signal_file, dtype=str).reset_index(drop=True)
        # Labels transposed: each row = one Psi coefficient vector
        self.label_df = pd.read_csv(label_file, dtype=str).T.reset_index(drop=True)
        self.len = min(len(self.signal_df), len(self.label_df))
        if max_samples:
            self.len = min(self.len, max_samples) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Convert signals + labels to complex arrays
        sig = self.signal_df.iloc[idx].apply(convert_to_complex).values.astype(np.complex64)
        lab = self.label_df.iloc[idx].apply(convert_to_complex).values.astype(np.complex64)

        # Normalize and split into [real, imag]
        sig = split_complex_to_imaginary(normalize_complex_to_unit_range(sig))
        lab = split_complex_to_imaginary(normalize_complex_to_unit_range(lab))

        return torch.tensor(sig, dtype=torch.float32), torch.tensor(lab, dtype=torch.float32)

# ------------------------------------------------------------------
#  Meta-data tensors (range, wavenumbers)
# ------------------------------------------------------------------
x_range = pd.read_csv(config["paths"]["x_range_file_path"]).iloc[50:-50, 0].values
x_range_tensor = torch.tensor(x_range, dtype=torch.float32).to(device)

setup = json.load(open(config["paths"]["setup_file_path"]))
F, ionoNHarm, xi = setup["F"], setup["ionoNharm"], setup["xi"]

# Harmonic wavenumbers
kpsi_values = pd.read_csv(config["paths"]["kpsi_file_path"], header=None).values.squeeze()
kpsi_tensor = torch.tensor(kpsi_values, dtype=torch.float32).to(device)

# Other constants
zero_pad         = 50
num_l4_samples   = 4
DX               = 0.25  # integration step size

# ------------------------------------------------------------------
#  Data loaders
# ------------------------------------------------------------------
train_dataset = LazySignalDataset(
    config["paths"]["signal_data_file_path"],
    config["paths"]["label_data_file_path"],
    max_samples=config["training"].get("max_train_samples")
)
test_dataset = LazySignalDataset(
    config["paths"]["test_data_file_path"],
    config["paths"]["test_label_file_path"],
    max_samples=config["training"].get("max_test_samples")
)

dloader_train = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
dloader_test  = DataLoader(test_dataset,  batch_size=config["training"]["batch_size"], shuffle=False)

# ------------------------------------------------------------------
#  Model
# ------------------------------------------------------------------
class ConfigurableModel(nn.Module):
    """
    Flexible feedforward neural network:
    - Architecture (layer sizes) specified in config
    - Uses dropout + chosen activation
    - Outputs 12 real numbers (6 real + 6 imaginary coefficients)
    """
    def __init__(self, arch, activation_fn, dropout_rate, input_dim, output_dim=12):
        super().__init__()
        layers, in_f = [], input_dim
        for size in arch:
            layers.extend([nn.Linear(in_f, size), nn.Dropout(dropout_rate), activation_fn()])
            in_f = size
        layers.append(nn.Linear(in_f, output_dim))  # final output layer
        self.seq = nn.Sequential(*layers)

    def forward(self, x):  # input shape [B, D]
        return self.seq(x)

# Instantiate model
inp_dim = train_dataset[0][0].numel()
activation_cls = getattr(nn, config["model"]["activation"])
dropout_rate   = config["model"].get("dropout_rate", 0.4)

model = ConfigurableModel(
    config["model"]["architecture"],
    activation_fn=activation_cls,
    dropout_rate=dropout_rate,
    input_dim=inp_dim
).to(device)

# Optimizer: AdamW (Adam with weight decay for regularization)
optimizer  = optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"]["fixed"],
    weight_decay=config["training"].get("l2_reg_weight", 0.0),
)

# ------------------------------------------------------------------
#  Loss weights (from config)
# ------------------------------------------------------------------
l4_weight        = config["training"].get("l4_weight", 0.0)
fourier_weight   = config["training"].get("fourier_weight", 0.0)
fourier_d1_weight= config["training"].get("fourier_d1_weight", 0.0)
fourier_d2_weight= config["training"].get("fourier_d2_weight", 0.0)

# ------------------------------------------------------------------
#  Training Loop
# ------------------------------------------------------------------
for epoch in tqdm(range(config["optimizer"]["maxiter_adam"]), desc="Epochs"):
    # ---------- Train ----------
    model.train()
    train_loss_epoch = 0.0

    for bx, by in tqdm(dloader_train, desc=f"Train {epoch+1}", leave=False):
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()

        # Forward pass: predict Psi coefficients
        preds       = model(bx)
        pr, pi      = preds[:, :6], preds[:, 6:]   # split into real/imag parts
        tr, ti      = by[:, :6], by[:, 6:]

        # Fourier-domain coefficient errors
        sq          = (pr - tr) ** 2 + (pi - ti) ** 2
        idx         = torch.arange(6, device=device, dtype=torch.float32)
        loss_dir    = torch.mean(torch.sum(sq, dim=1))          # direct MSE
        loss_d1     = torch.mean(torch.sum((idx**2) * sq, dim=1))  # weighted by k^2
        loss_d2     = torch.mean(torch.sum((idx**4) * sq, dim=1))  # weighted by k^4

        # Image-domain L4 loss (subsampled)
        loss_l4     = calculate_l4_batch_subsample(
            bx, pr, pi, x_range_tensor, kpsi_tensor, F, DX, xi,
            zero_pad, num_samples=num_l4_samples
        )

        # Weighted sum of loss terms
        total = (
            fourier_weight   * loss_dir +
            fourier_d1_weight* loss_d1   +
            fourier_d2_weight* loss_d2   +
            l4_weight        * loss_l4
        )
        total.backward()
        optimizer.step()
        train_loss_epoch += total.item()

    # ---------- Test ----------
    model.eval()
    test_loss_epoch = 0.0
    with torch.no_grad():
        for bx_t, by_t in dloader_test:
            bx_t, by_t = bx_t.to(device), by_t.to(device)
            preds_t    = model(bx_t)
            pr_t, pi_t = preds_t[:, :6], preds_t[:, 6:]
            tr_t, ti_t = by_t[:, :6], by_t[:, 6:]

            # Fourier-domain errors
            sq_t   = (pr_t - tr_t) ** 2 + (pi_t - ti_t) ** 2
            idx_t  = torch.arange(6, device=device, dtype=torch.float32)
            loss_dir_t = torch.mean(torch.sum(sq_t, dim=1))
            loss_d1_t  = torch.mean(torch.sum((idx_t**2) * sq_t, dim=1))
            loss_d2_t  = torch.mean(torch.sum((idx_t**4) * sq_t, dim=1))

            # Image-domain L4 loss
            loss_l4_t = calculate_l4_batch_subsample(
                bx_t, pr_t, pi_t, x_range_tensor, kpsi_tensor, F, DX, xi,
                zero_pad, num_samples=num_l4_samples
            )

            total_t = (
                fourier_weight    * loss_dir_t +
                fourier_d1_weight * loss_d1_t  +
                fourier_d2_weight * loss_d2_t  +
                l4_weight         * loss_l4_t
            )
            test_loss_epoch += total_t.item()

    # ---------- Log ----------
    print(f"Epoch {epoch+1}: train={train_loss_epoch:.6f} | test={test_loss_epoch:.6f}")
    with open("training_losses_torch.csv", "a") as f:
        f.write(f"{epoch+1},{train_loss_epoch},{test_loss_epoch}\n")

# ------------------------------------------------------------------
#  Save final weights
# ------------------------------------------------------------------
with open("model_weights_torch.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

print("Training complete.")
