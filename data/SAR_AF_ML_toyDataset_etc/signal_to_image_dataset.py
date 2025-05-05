import os
import json
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# --- Configuration ---
DATA_DIR = "/home/houtlaw/iono-net/data/baselines/10k_lownoise"
X_RANGE_PATH = f"{DATA_DIR}/meta_X_20250206_104914.csv"
SETUP_PATH = f"{DATA_DIR}/setup_20250206_104914.json"
SIGNAL_PATH = f"/home/houtlaw/iono-net/data/baselines/10k_lownoise/train_uscStruct_vals_20250206_104902.csv"
PSI_PATH = f"/home/houtlaw/iono-net/data/baselines/10k_lownoise/train_compl_ampls_20250206_104902.csv"
KPSI_PATH = f"{DATA_DIR}/kPsi_20250206_104914.csv"
OUTPUT_PATH_FOCUSED = f"{DATA_DIR}/train_image_recon_jnp.csv"
OUTPUT_PATH_UNFOCUSED = f"{DATA_DIR}/train_image_recon_jnp_unfocused.csv"
X_TRIM_PATH = f"{DATA_DIR}/x_range_image_recon_jnp.csv"

# --- Helpers ---
def convert_to_complex(s):
    s = str(s)
    if s == "NaNNaNi":
        return np.nan
    return complex(s.replace('i', 'j'))

def jnp_image_reconstruction(x_range, signal_vals, pr, pi, kpsi_values, F, dx, xi):
    F2 = F / 2
    mask = (x_range >= (x_range[0] + F2)) & (x_range <= (x_range[-1] - F2))
    x_trimmed = x_range[mask]
    signal_trimmed = signal_vals[mask]
    offsets = jnp.linspace(-F2, F2, int(F/dx)+1)
    
    def trapz_nonuniform_jax(y, x):
        dx = x[1:] - x[:-1]
        avg_y = 0.5 * (y[1:] + y[:-1])
        return jnp.sum(dx * avg_y)

    def eval_point(y):
        base = y + offsets
        interp = jnp.interp(base, x_trimmed, signal_trimmed)
        waveform = jnp.exp(-1j * jnp.pi * (base - y) ** 2 / F)
        sarr = xi * base + (1 - xi) * y
        psi_pred = jnp.exp(1j * (
            jnp.sum(pr[:, None] * jnp.cos(jnp.outer(sarr, kpsi_values).T), axis=0) +
            jnp.sum(pi[:, None] * jnp.sin(jnp.outer(sarr, kpsi_values).T), axis=0)
        ))
        integrand = waveform * interp * psi_pred
        return trapz_nonuniform_jax(integrand, base) / F

    return jax.vmap(eval_point)(x_trimmed), x_trimmed

# --- Load setup and data ---
x_range = pd.read_csv(X_RANGE_PATH).iloc[:, 0].values
with open(SETUP_PATH) as f:
    setup = json.load(f)
F, xi, DX = setup["F"], setup["xi"], 0.25
kpsi_values = pd.read_csv(KPSI_PATH).values.flatten()

signal_df = pd.read_csv(SIGNAL_PATH, dtype=str).map(convert_to_complex).T
psi_df = pd.read_csv(PSI_PATH, dtype=str).map(lambda s: complex(s.replace('i', 'j'))).T

assert signal_df.shape[0] == psi_df.shape[0]

# --- Process all samples (focused and unfocused) ---
image_dataset_focused = []
image_dataset_unfocused = []
x_final = None

for i in tqdm(range(signal_df.shape[0]), desc="Processing signals"):
    signal_vals = jnp.array(signal_df.iloc[i].values)
    psi_coeffs = psi_df.iloc[i].values
    pr = jnp.real(psi_coeffs)
    pi = -jnp.imag(psi_coeffs)
    
    image_focused, x_used = jnp_image_reconstruction(x_range, signal_vals, pr, pi, kpsi_values, F, DX, xi)
    image_unfocused, _ = jnp_image_reconstruction(x_range, signal_vals, jnp.zeros_like(pr), jnp.zeros_like(pi), kpsi_values, F, DX, xi)

    if x_final is None:
        x_final = np.array(x_used)

    image_dataset_focused.append(np.array(image_focused))
    image_dataset_unfocused.append(np.array(image_unfocused))

# --- Save datasets ---
pd.DataFrame(np.stack(image_dataset_focused)).to_csv(OUTPUT_PATH_FOCUSED, index=False)
pd.DataFrame(np.stack(image_dataset_unfocused)).to_csv(OUTPUT_PATH_UNFOCUSED, index=False)
pd.DataFrame(x_final).to_csv(X_TRIM_PATH, index=False, header=False)

print(f"Saved focused image dataset to: {OUTPUT_PATH_FOCUSED}")
print(f"Saved unfocused image dataset to: {OUTPUT_PATH_UNFOCUSED}")
print(f"Saved trimmed x-range to: {X_TRIM_PATH}")
