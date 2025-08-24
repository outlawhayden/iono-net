# === PyTorch 1-D U-Net Training Script: Focused-Image Prediction (residual identity init) ===
# -------------------------------------------------------------------------------------------
# Purpose:
#   - Train a 1D U-Net to map unfocused SAR/iono-net image signals → focused image signals.
#   - Uses residual identity initialization: model starts as near-identity mapping,
#     so network learns corrections rather than the entire signal.
#   - Adds negative L4 sparsity term to encourage sharp, peaked reconstructions.
#   - Saves training/validation losses and final model weights for later evaluation.
# -------------------------------------------------------------------------------------------

import os, time, pickle, json
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------
# Load YAML config with paths, training hyperparameters, etc.
with open("config_focus.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Save config to CSV for reproducibility (audit trail)
with open("config_used.csv", "w") as f:
    for k, v in cfg.items():
        f.write(f"{k},{v}\n")

# Initialize CSV log of training and validation losses
with open("training_losses_focus.csv", "w") as f:
    f.write("Epoch,TrainLoss,ValLoss\n")

# Select GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}]  Using device: {device}")

# ------------------------------------------------------------------
#  Helper utilities for preprocessing
# ------------------------------------------------------------------
def _to_complex64(series):
    """
    Convert CSV string → complex64.
    Handles MATLAB placeholder 'NaNNaNi' by mapping to 0j.
    """
    s = str(series)
    return 0j if s == "NaNNaNi" else complex(s.replace("i", "j"))

def _norm_to_unit(v):
    """
    Normalize vector to unit amplitude max.
    Preserves scale invariance between different samples.
    """
    m = np.abs(v).max()
    return v if m == 0 else v / m

def _split_real_imag(a):
    """
    Flatten complex array into concatenated real+imag channels.
    Example: [a+ib, c+id] → [a, c, b, d].
    """
    return np.concatenate([a.real, a.imag], axis=-1)

# ------------------------------------------------------------------
#  Lazy CSV Dataset
# ------------------------------------------------------------------
class LazyImageDataset(Dataset):
    """
    Custom Dataset that:
      - Reads unfocused and focused image CSVs (lazy row access).
      - Each sample is a complex-valued sequence (unfocused → input, focused → target).
      - Normalizes each sample independently to unit amplitude.
      - Returns tensors with real+imag split into two channels.
    """
    def __init__(self, unfocused_csv, focused_csv, max_samples=None):
        # Note: Transpose (T) makes each *row* correspond to a sample
        self.unfocus = pd.read_csv(unfocused_csv, dtype=str).T.reset_index(drop=True)
        self.focus   = pd.read_csv(focused_csv,   dtype=str).T.reset_index(drop=True)
        self.length  = min(len(self.unfocus), len(self.focus))
        if max_samples:
            self.length = min(self.length, max_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Convert string row → complex64
        u = self.unfocus.iloc[idx].apply(_to_complex64).values.astype(np.complex64)
        f = self.focus.iloc[idx].apply(_to_complex64).values.astype(np.complex64)
        # Normalize amplitude
        u, f = _norm_to_unit(u), _norm_to_unit(f)
        # Convert to split real+imag and return as torch tensors
        return torch.from_numpy(_split_real_imag(u).astype(np.float32)), \
               torch.from_numpy(_split_real_imag(f).astype(np.float32))

# ------------------------------------------------------------------
#  1-D U-Net Module with Residual Output
# ------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Small building block: Conv1d + BatchNorm + ReLU."""
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__(nn.Conv1d(c_in, c_out, k, padding=p, bias=False),
                         nn.BatchNorm1d(c_out),
                         nn.ReLU(inplace=True))

class UNet1D(nn.Module):
    """
    1-D U-Net:
      - Encoder: downsampling conv blocks with max-pooling.
      - Bottleneck: deeper conv block at lowest resolution.
      - Decoder: upsampling with skip connections from encoder.
      - Head: 1x1 convolution to map back to 2 channels (real, imag).
      - Residual output: returns input + learned residual (near identity).
    """
    def __init__(self, base_ch=32, depth=4, in_ch=2, out_ch=2, dropout=0.0):
        super().__init__()
        self.depth = depth

        # Encoder pathway
        enc, dec, pools, ups = [], [], [], []
        ch = in_ch
        for d in range(depth):
            enc.append(ConvBNReLU(ch, base_ch * 2**d))
            pools.append(nn.MaxPool1d(2))
            ch = base_ch * 2**d

        # Bottleneck block (deepest layer)
        self.bottle = ConvBNReLU(ch, ch * 2)
        ch *= 2

        # Decoder pathway with skip connections
        for d in reversed(range(depth)):
            ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec.append(ConvBNReLU(ch + base_ch * 2**d, base_ch * 2**d))
            ch = base_ch * 2**d

        # Register modules
        self.enc, self.pools, self.ups, self.dec = map(nn.ModuleList, (enc, pools, ups, dec))
        self.head = nn.Conv1d(ch, out_ch, 1)  # final 1x1 conv
        self.drop = nn.Dropout(dropout)

    def forward(self, x_flat):
        # Reshape [B, 2L] → [B, 2, L] (two channels: real, imag)
        B, F = x_flat.shape
        L = F // 2
        x = x_flat.view(B, 2, L)

        # --- Encoder ---
        skips = []
        for enc, pool in zip(self.enc, self.pools):
            x = enc(x)
            skips.append(x)   # save skip connection
            x = pool(x)

        # --- Bottleneck ---
        x = self.bottle(x)

        # --- Decoder ---
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            # Align spatial length with skip (handle odd sizes)
            if x.shape[-1] != skip.shape[-1]:
                diff = x.shape[-1] - skip.shape[-1]
                x = x[..., :-diff] if diff > 0 else torch.nn.functional.pad(x, (0, -diff))
            # Concatenate skip connection
            x = dec(torch.cat([x, skip], dim=1))
            x = self.drop(x)

        # --- Residual head ---
        residual = self.head(x).view(B, -1)
        return x_flat + residual  # Identity + residual correction

# ------------------------------------------------------------------
#  Data loaders
# ------------------------------------------------------------------
train_ds = LazyImageDataset(cfg['paths']['unfocused_train'],
                            cfg['paths']['focused_train'],
                            cfg['training'].get('max_train_samples'))
val_ds   = LazyImageDataset(cfg['paths']['unfocused_val'],
                            cfg['paths']['focused_val'],
                            cfg['training'].get('max_val_samples'))

train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                          shuffle=True,  num_workers=cfg['training'].get('num_workers', 0))
val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                          shuffle=False, num_workers=cfg['training'].get('num_workers', 0))

# ------------------------------------------------------------------
#  Model, Optimizer, Loss
# ------------------------------------------------------------------
depth = cfg['model'].get('depth', 4)
model = UNet1D(base_ch=cfg['model']['base_channels'],
               depth=depth,
               dropout=cfg['model'].get('dropout_rate', 0.2)).to(device)

# === Initialize model near identity ===
#   - Convs: He initialization (for ReLU stability).
#   - Head: zeros (ensures initial output ≈ identity mapping).
def init_he(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_he)
nn.init.zeros_(model.head.weight)
nn.init.zeros_(model.head.bias)

# Optimizer = AdamW (Adam + weight decay regularization)
optimizer = optim.AdamW(model.parameters(),
                        lr=cfg['optimizer']['lr'],
                        weight_decay=cfg['optimizer'].get('weight_decay', 0.0))

# Loss function: mean squared error (MSE) on predicted vs. focused images
criterion = nn.MSELoss()

# ---------------------------------------------------------------
#   L4 sparsity term
# ---------------------------------------------------------------
def negative_l4_norm(pred_flat):
    """
    Compute negative L4 norm of predicted amplitudes:
      - Encourages sparse, peaked outputs (better focus).
      - Negative sign means optimizer maximizes peakedness.
    """
    B, F = pred_flat.shape
    L = F // 2
    real, imag = pred_flat[:, :L], pred_flat[:, L:]
    amp4 = (real**2 + imag**2) ** 2
    return -torch.mean(torch.sum(amp4, dim=1))

# ------------------------------------------------------------------
#  Forward with padding and cropping
# ------------------------------------------------------------------
def forward_with_padding(xb):
    """
    Ensures input length divisible by 2^depth for U-Net downsampling.
    Pads to nearest multiple of 2^depth, applies model, then crops back.
    """
    m = 2 ** depth
    pad = (-xb.shape[1]) % m
    if pad:
        xb = torch.nn.functional.pad(xb, (0, pad))
    preds = model(xb)
    if pad:
        preds = preds[:, :-pad]
    return preds

# Weight of L4 loss term (from config)
lambda_l4 = cfg['loss']['l4_weight']

# ------------------------------------------------------------------
#  Training loop
# ------------------------------------------------------------------
for epoch in range(1, cfg['training']['epochs'] + 1):
    # --- Training phase ---
    model.train()
    run_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = forward_with_padding(xb)
        mse   = criterion(preds, yb)
        l4    = negative_l4_norm(preds)
        scaled_l4 = l4 / (1 + l4)  # squash for stability
        loss  = mse + lambda_l4 * scaled_l4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()
        run_loss += loss.item()

    # --- Validation phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = forward_with_padding(xb)
            mse_val = criterion(preds, yb)
            l4_val  = negative_l4_norm(preds)
            val_loss += (mse_val + lambda_l4 * l4_val).item()

    # --- Logging ---
    print(f"Epoch {epoch:03d} | train {run_loss:.6f} | val {val_loss:.6f}")
    with open("training_losses_focus.csv", "a") as f:
        f.write(f"{epoch},{run_loss},{val_loss}\n")

# ------------------------------------------------------------------
#  Save weights
# ------------------------------------------------------------------
with open("unet_focus_weights.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

print("Finished training.")
