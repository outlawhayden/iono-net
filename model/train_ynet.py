import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# === Load Config ===
with open("config_ynet.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("config_used.csv", "w") as f:
    for key, val in config.items():
        f.write(f"{key},{val}\n")

with open("training_losses_ynet.csv", "w") as f:
    f.write("Epoch,TrainLoss,TestLoss\n")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Helper Functions ===
def convert_to_complex(s):
    if s == "NaNNaNi":
        return 0
    return complex(s.replace('i', 'j'))

def add_gaussian_noise(x, std=0.01):
    return x + std * torch.randn_like(x)

def normalize_complex_to_unit_range(matrix):
    amp = np.abs(matrix)
    amp_max = np.max(amp, axis=1, keepdims=True)
    amp_max[amp_max == 0] = 1
    return matrix / amp_max

def split_complex_to_imaginary(complex_array):
    return np.concatenate([complex_array.real, complex_array.imag], axis=-1)

# === Model ===
class YNetModel(nn.Module):
    def __init__(self, shared_arch, psi_head_arch, image_head_arch,
                 activation_fn=nn.ReLU, dropout_rate=0.2,
                 input_dim=2882, psi_out_dim=12, image_out_dim=2882):
        super().__init__()
        layers = []
        in_features = input_dim
        for size in shared_arch:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation_fn())
            in_features = size
        self.shared = nn.Sequential(*layers)

        psi_layers = []
        psi_in = in_features
        for size in psi_head_arch:
            psi_layers.append(nn.Linear(psi_in, size))
            psi_layers.append(nn.Dropout(dropout_rate))
            psi_layers.append(activation_fn())
            psi_in = size
        psi_layers.append(nn.Linear(psi_in, psi_out_dim))
        self.psi_head = nn.Sequential(*psi_layers)

        image_layers = []
        image_in = in_features
        for size in image_head_arch:
            image_layers.append(nn.Linear(image_in, size))
            image_layers.append(nn.Dropout(dropout_rate))
            image_layers.append(activation_fn())
            image_in = size
        image_layers.append(nn.Linear(image_in, image_out_dim))
        self.image_head = nn.Sequential(*image_layers)

    def forward(self, x):
        shared = self.shared(x)
        return self.psi_head(shared), self.image_head(shared)

# === Load Training Data ===
def load_data(path_dict, device):
    def load_csv_as_complex(path):
        df = pd.read_csv(path, dtype=str)
        mat = df.map(convert_to_complex).to_numpy().T
        return normalize_complex_to_unit_range(mat)
    
    def load_csv_as_complex_rows(path):
        df = pd.read_csv(path, dtype=str)
        mat = df.map(convert_to_complex).to_numpy()  # no transpose
        return normalize_complex_to_unit_range(mat)


    input_data = split_complex_to_imaginary(load_csv_as_complex(path_dict['signal_data_file_path']))
    coeff_data = split_complex_to_imaginary(load_csv_as_complex(path_dict['label_data_file_path']))
    image_data = split_complex_to_imaginary(load_csv_as_complex_rows(path_dict['focused_image_file_path']))

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    coeff_tensor = torch.tensor(coeff_data, dtype=torch.float32).to(device)
    image_tensor = torch.tensor(image_data, dtype=torch.float32).to(device)


    print("Input tensor shape:", input_tensor.shape)
    print("Coeff tensor shape:", coeff_tensor.shape)
    print("Image tensor shape:", image_tensor.shape)


    return TensorDataset(input_tensor, coeff_tensor, image_tensor)

train_dataset = load_data(config['paths'], device)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

# === Load Test Data ===
def load_test_data(path_dict, device):
    test_paths = {
        'signal_data_file_path': path_dict['test_data_file_path'],
        'label_data_file_path': path_dict['test_label_file_path'],
        'focused_image_file_path': path_dict['test_focused_image_file_path']
    }
    return load_data(test_paths, device)

test_dataset = load_test_data(config['paths'], device)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# === Instantiate Model ===
input_dim = train_dataset[0][0].shape[0]
activation_cls = getattr(nn, config['model']['activation'])
dropout_rate = config['model'].get('dropout_rate', 0.2)

model = YNetModel(
    shared_arch=config['model']['architecture'],
    psi_head_arch=config['model'].get('psi_head_arch', [64]),
    image_head_arch=config['model'].get('image_head_arch', [128]),
    activation_fn=activation_cls,
    dropout_rate=dropout_rate,
    input_dim=input_dim,
    psi_out_dim=12,
    image_out_dim=1882
).to(device)

# === Optimizer ===
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['learning_rate']['fixed'],
    weight_decay=config['training'].get('l2_reg_weight', 0.0)
)

# === Loss Weights ===
coeff_weight = config['training'].get('coeff_loss_weight', 1.0)
image_weight = config['training'].get('image_loss_weight', 1.0)

# === Training Loop ===
for epoch in tqdm(range(config['optimizer']['maxiter_adam'])):
    model.train()
    total_loss = 0.0

    for batch_x, batch_coeff, batch_image in train_loader:
        optimizer.zero_grad()
        noisy_input = add_gaussian_noise(batch_x, std=0.1)
        pred_coeff, pred_image = model(noisy_input)

        coeff_loss = nn.functional.mse_loss(pred_coeff, batch_coeff)
        image_loss = nn.functional.mse_loss(pred_image, batch_image)
        loss = coeff_weight * coeff_loss + image_weight * image_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # === Evaluate ===
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for test_x, test_coeff, test_image in test_loader:
            pred_coeff, pred_image = model(test_x)
            coeff_loss = nn.functional.mse_loss(pred_coeff, test_coeff) # mse loss of Ψ coefficients
            image_loss = nn.functional.mse_loss(pred_image, test_image) # mse loss between image head and true focused image
            test_loss += (coeff_weight * coeff_loss + image_weight * image_loss).item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.6f}, Test Loss: {test_loss:.6f}")
    with open("training_losses_ynet.csv", "a") as f:
        f.write(f"{epoch},{total_loss},{test_loss}\n")

# === Save Model ===
with open("model_weights_ynet.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

print("Training complete.")
