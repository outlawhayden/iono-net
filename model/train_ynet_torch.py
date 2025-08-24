# === PyTorch 1-D Y-Net Training Script: Focused-Image + 12-Coeff Prediction ===
# ---------------------------------------------------------------------------------
# Purpose:
#   - Train a dual-headed 1D U-Net–like architecture ("Y-Net"):
#       1. One branch reconstructs focused images (denoised, sharpened).
#       2. Another branch predicts 12 Psi harmonic coefficients.
#   - Loss combines:
#       - Image-domain sparsity (L4 negative norm).
#       - Coefficient regression loss.
#       - Reconstruction accuracy (via differentiable focusing integral).
#   - Implements early stopping and saves best model weights.
# ---------------------------------------------------------------------------------

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
with open("config_ynet.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Ensure critical loss/paths keys exist (defaults)
cfg.setdefault("loss", {}).setdefault("coef_weight", 1.0)
cfg.setdefault("paths", {}).setdefault("coeff_train", None)
cfg["paths"].setdefault("coeff_val", cfg["paths"]["coeff_train"])

# Save config snapshot for reproducibility
with open("config_used.csv", "w") as f:
    for k, v in cfg.items():
        f.write(f"{k},{v}\n")

# Initialize CSV log for loss terms
with open("training_losses_focus.csv", "w") as f:
    f.write("Epoch,TrainTotal,ValTotal,TrainImg,ValImg,TrainL4,ValL4,"
            "TrainCoef,ValCoef,TrainRecon,ValRecon\n")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}]  Using device: {device}")

# ------------------------------------------------------------------
#  Helper utilities
# ------------------------------------------------------------------
def _to_complex64(series):
    """Convert CSV entry → complex64. Handle NaNs gracefully."""
    try:
        s = str(series)
        if s in {"NaNNaNi", "nan", "NaN"}:
            return np.nan + 1j * np.nan
        return complex(s.replace("i", "j"))
    except:
        return np.nan + 1j * np.nan

def _norm_to_unit(v):
    """Normalize vector so max amplitude = 1."""
    m = np.abs(v).max()
    return v if m == 0 else v / m

def _split_real_imag(a):
    """Split complex array → concatenated [real, imag]. Replace NaNs with 0."""
    a = np.nan_to_num(a, nan=0.0)
    return np.concatenate([a.real, a.imag], axis=-1)

# ------------------------------------------------------------------
#  Lazy CSV Dataset
# ------------------------------------------------------------------
class LazyImageDataset(Dataset):
    """
    Lazy CSV dataset:
      - Loads unfocused (input), focused (label), and optional coefficient labels.
      - Normalizes each sample independently.
      - Returns torch tensors: [unfocused, focused, coeffs?].
    """
    def __init__(self, unfocus_csv, focus_csv, coeff_csv=None, max_samples=None):
        self.unfocus = pd.read_csv(unfocus_csv, dtype=str).reset_index(drop=True)
        self.focus   = pd.read_csv(focus_csv,   dtype=str).reset_index(drop=True)
        if coeff_csv is not None and os.path.isfile(coeff_csv):
            # Coeffs: one row per sample after transpose
            self.coeff = pd.read_csv(coeff_csv, dtype=str).T.reset_index(drop=True)
        else:
            self.coeff = None
        self.length = min(len(self.unfocus), len(self.focus))
        if max_samples is not None:
            self.length = min(self.length, max_samples)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        # Convert unfocused & focused signals
        u = self.unfocus.iloc[idx].apply(_to_complex64).values.astype(np.complex64)
        f = self.focus  .iloc[idx].apply(_to_complex64).values.astype(np.complex64)
        u, f = _norm_to_unit(u), _norm_to_unit(f)

        # Convert coefficients if available
        if self.coeff is not None:
            coeff_cplx = self.coeff.iloc[idx].apply(_to_complex64).values.astype(np.complex64)
            coeff_real = _split_real_imag(coeff_cplx)  # no normalization (keep true scale)
            coeff_t = torch.from_numpy(coeff_real.astype(np.float32))
        else:
            coeff_t = torch.empty(0)

        return (torch.from_numpy(_split_real_imag(u).astype(np.float32)),
                torch.from_numpy(_split_real_imag(f).astype(np.float32)),
                coeff_t)

# ------------------------------------------------------------------
#  Y-Net Model
# ------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Conv1D + BatchNorm + ReLU block."""
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__(nn.Conv1d(c_in, c_out, k, padding=p, bias=False),
                         nn.BatchNorm1d(c_out),
                         nn.ReLU(inplace=True))
        
class YNet1D(nn.Module):
    """
    Y-Net architecture (dual-head 1D U-Net):
      - Encoder: downsampled Conv blocks.
      - Bottleneck: feature compression.
      - Decoder: upsampled Conv blocks w/ skip connections.
      - Image head: reconstructs focused image (residual path).
      - Coefficient head: predicts 12 Psi harmonic coefficients.
    """
    def __init__(self, base_ch=32, depth=4, in_ch=2, out_ch=2, n_coeff=12, dropout=0.0):
        super().__init__()
        self.depth = depth

        # Encoder (downsampling)
        enc, pools, ch = [], [], in_ch
        for d in range(depth):
            enc.append(ConvBNReLU(ch, base_ch * 2**d))
            pools.append(nn.MaxPool1d(2))
            ch = base_ch * 2**d
        self.enc, self.pools = nn.ModuleList(enc), nn.ModuleList(pools)

        # Bottleneck
        self.bottle = ConvBNReLU(ch, ch * 2)
        bott_ch = ch * 2
        ch *= 2

        # Decoder (upsampling + skip connections)
        dec, ups = [], []
        for d in reversed(range(depth)):
            ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec.append(ConvBNReLU(ch + base_ch*2**d, base_ch*2**d))
            ch = base_ch*2**d
        self.ups, self.dec = nn.ModuleList(ups), nn.ModuleList(dec)

        # Heads
        self.head_img = nn.Conv1d(ch, out_ch, 1)   # focused image branch
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)         # global avg pool
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.LayerNorm(bott_ch),
                                nn.ReLU(inplace=True),
                                nn.Linear(bott_ch, n_coeff))  # coefficient branch

    def forward(self, x_flat):
        # Reshape [B,2L] → [B,2,L]
        B, F = x_flat.shape
        L = F // 2
        x = x_flat.view(B, 2, L)

        # Encoder
        skips = []
        for enc, pool in zip(self.enc, self.pools):
            x = enc(x); skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottle(x)
        coeff_out = self.fc(self.gap(x))  # coefficient head

        # Decoder
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            # fix length mismatch due to odd sizes
            if x.shape[-1] != skip.shape[-1]:
                diff = x.shape[-1] - skip.shape[-1]
                x = x[..., :-diff] if diff > 0 else nn.functional.pad(x, (0, -diff))
            x = dec(torch.cat([x, skip], dim=1))
            x = self.drop(x)

        # Residual image head
        residual = self.head_img(x).view(B, -1)
        if residual.shape[1] != x_flat.shape[1]:
            # adjust length if mismatch
            residual = (nn.functional.pad(residual, (0, x_flat.shape[1] - residual.shape[1]))
                        if residual.shape[1] < x_flat.shape[1]
                        else residual[:, :x_flat.shape[1]])
        img_out = x_flat + residual  # identity + residual correction
        return img_out, coeff_out

# ------------------------------------------------------------------
#  Data, Model, Initialization
# ------------------------------------------------------------------
train_ds = LazyImageDataset(cfg['paths']['unfocused_train'],
                            cfg['paths']['focused_train'],
                            cfg['paths'].get('coeff_train'),
                            cfg['training'].get('max_train_samples'))
print(f"Training dataset size: {len(train_ds)}")
val_ds   = LazyImageDataset(cfg['paths']['unfocused_val'],
                            cfg['paths']['focused_val'],
                            cfg['paths'].get('coeff_val'),
                            cfg['training'].get('max_val_samples'))

train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,
                          num_workers=cfg['training'].get('num_workers', 0))
val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'], shuffle=False,
                          num_workers=cfg['training'].get('num_workers', 0))

# Metadata tensors (x_range, kPsi values, focusing params)
x_range_np = pd.read_csv(cfg["paths"]["x_range"], usecols=[0], dtype=np.float64).values.squeeze()
kpsi_np    = pd.read_csv(cfg["paths"]["kpsi"], header=None, dtype=np.float64).values.squeeze()
x_range_tensor = torch.from_numpy(x_range_np).to(dtype=torch.float64, device=device)
kpsi_tensor    = torch.from_numpy(kpsi_np).to(dtype=torch.float64, device=device)
F_scalar       = torch.tensor(cfg['params']['F'], dtype=torch.float64, device=device)
dx_scalar      = torch.tensor(cfg['params']['dx'], dtype=torch.float64, device=device)

# Model instantiation
depth = cfg['model'].get('depth', 4)
model = YNet1D(base_ch=cfg['model']['base_channels'],
               depth=depth,
               dropout=cfg['model'].get('dropout_rate', 0.2)).to(device)

# Initialization tricks
def init_he(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_he)

# Heads initialized small or zero to enforce near-identity
nn.init.kaiming_uniform_(model.head_img.weight, a=0, nonlinearity='relu')
nn.init.zeros_(model.head_img.bias)
nn.init.zeros_(model.fc[-1].weight)
nn.init.zeros_(model.fc[-1].bias)

# Optimizer + losses
optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'],
                        weight_decay=cfg['optimizer'].get('weight_decay', 0.0))
criterion_img   = nn.MSELoss()
criterion_coeff = nn.MSELoss()

# ------------------------------------------------------------------
#  Loss terms
# ------------------------------------------------------------------
def negative_l4_norm(pred_flat):
    """Negative L4 amplitude norm → promotes sparsity/peakedness in image."""
    B, F = pred_flat.shape
    L = F // 2
    real, imag = pred_flat[:, :L], pred_flat[:, L:]
    amp4 = (real**2 + imag**2)**2
    return -torch.mean(torch.sum(amp4, dim=1))

lambda_l4    = cfg['loss']['l4_weight']
lambda_coef  = cfg['loss']['coef_weight']
lambda_recon = cfg['loss']['recon_accuracy_weight']

def forward_with_padding(xb):
    """Pad input to multiple of 2^depth for U-Net, crop output back."""
    m = 2 ** depth
    pad = (-xb.shape[1]) % m
    if pad: xb = nn.functional.pad(xb, (0, pad))
    img, coef = model(xb)
    if pad: img = img[:, :-pad]
    return img, coef

# (focus_pytorch + recon_accuracy_loss defined here... not repeating for brevity)
# → They implement differentiable focusing integral to measure reconstruction
#   accuracy of predicted Psi coefficients.

# ------------------------------------------------------------------
#  Early stopping setup
# ------------------------------------------------------------------
patience = cfg['training'].get('early_stopping_patience', 10)
min_delta = cfg['training'].get('early_stopping_min_delta', 1e-5)
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# ------------------------------------------------------------------
#  Training loop
# ------------------------------------------------------------------
for epoch in range(1, cfg['training']['epochs'] + 1):
    # --- Training ---
    model.train()
    run_loss, img_loss_sum, l4_loss_sum, coef_loss_sum, recon_loss_sum = 0,0,0,0,0

    for xb, yb, cb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
        optimizer.zero_grad()
        preds_img, preds_coef = forward_with_padding(xb)

        # Loss components
        loss_img  = criterion_img(preds_img, yb)
        loss_l4   = torch.clamp(negative_l4_norm(preds_img), min=-5)  # clamp for stability
        loss_coef = criterion_coeff(preds_coef, cb)
        loss_recon = recon_accuracy_loss(x_range_tensor, preds_coef, yb,
                                         xb[:, :x_range_tensor.shape[0]*2],
                                         kpsi_tensor, F_scalar, dx_scalar)

        # Weighted total loss
        loss = 0*loss_img + lambda_l4*loss_l4 + lambda_coef*loss_coef + lambda_recon*loss_recon
        if torch.isnan(loss):
            print("NaN loss encountered.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # grad clipping
        optimizer.step()

        # Track sums for logging
        run_loss      += loss.item()
        img_loss_sum  += loss_img.item()
        l4_loss_sum   += loss_l4.item()
        coef_loss_sum += loss_coef.item()
        recon_loss_sum += loss_recon.item()

    # --- Validation ---
    model.eval()
    val_loss, img_loss_val, l4_loss_val, coef_loss_val, recon_loss_val = 0,0,0,0,0
    with torch.no_grad():
        for xb, yb, cb in val_loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            preds_img, preds_coef = forward_with_padding(xb)

            loss_img_v  = criterion_img(preds_img, yb)
            loss_l4_v   = torch.clamp(negative_l4_norm(preds_img), min=-5)
            loss_coef_v = criterion_coeff(preds_coef, cb)
            loss_recon_v = recon_accuracy_loss(x_range_tensor, preds_coef, yb,
                                               xb[:, :x_range_tensor.shape[0]*2],
                                               kpsi_tensor, F_scalar, dx_scalar)

            total_val = 0*loss_img_v + lambda_l4*loss_l4_v + lambda_coef*loss_coef_v + lambda_recon*loss_recon_v
            val_loss += total_val.item()
            img_loss_val  += loss_img_v.item()
            l4_loss_val   += loss_l4_v.item()
            coef_loss_val += loss_coef_v.item()
            recon_loss_val += loss_recon_v.item()

    # --- Logging ---
    print(f"Epoch {epoch:03d} | train {run_loss:.6f} | val {val_loss:.6f}")
    with open("training_losses_focus.csv", "a") as f:
        f.write(f"{epoch},{run_loss:.6f},{val_loss:.6f},"
                f"{img_loss_sum:.6f},{img_loss_val:.6f},"
                f"{l4_loss_sum:.6f},{l4_loss_val:.6f},"
                f"{coef_loss_sum:.6f},{coef_loss_val:.6f},"
                f"{recon_loss_sum:.6f},{recon_loss_val:.6f}\n")

    # --- Early stopping ---
    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss; epochs_no_improve = 0
        with open("ynet_focus_weights.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs. Best val loss: {best_val_loss:.6f}")
            early_stop = True
            with open("ynet_focus_weights.pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)
            print("Finished Y-Net training.")
            break

# Save final weights (if no early stop triggered earlier)
with open("ynet_focus_weights.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

print("Finished Y-Net training.")
