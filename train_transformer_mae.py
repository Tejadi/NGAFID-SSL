#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import pandas as pd
import glob
import wandb
from torch.utils.data import DataLoader
from datasets.masked_flight_dataset import MaskedFlightDataset
from models.transformer_autoencoder import TransformerAutoencoder
from utils import load_config

# Load configuration
config = load_config()
SS_PATH = config['paths']['flight_scores']
FLT_PATH = config['paths']['fixed_keys_flights']

# Hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 5

wandb.init(
    # set the wandb project where this run will be logged
    project="ngafid-ssl-fall-24",
    entity="ngafid-ssl",
    name=f'Transformer bs={batch_size}, lr={learning_rate}, n_epochs={num_epochs}.',

    # track hyperparameters and run metadata
    config={
        'learning_rate': learning_rate,
        'epochs': num_epochs,
    }
)

files = glob.glob(os.path.join(FLT_PATH, "*.csv"))
flight_paths_df = pd.DataFrame(files, columns=['file_path'])

dataset = MaskedFlightDataset(flight_paths_df, masking_ratio=0.6, mean_mask_length=3)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = TransformerAutoencoder(input_dim=44, d_model=128, num_heads=8,
                               num_encoder_layers=4, num_decoder_layers=4,
                               dim_feedforward=256, dropout=0.1).to(device)

# param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(param_count)
# breakpoint()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for (X_masked, X_orig, mask) in train_loader:
        X_masked = X_masked.to(device)    
        X_orig = X_orig.to(device)        
        mask = mask.to(device)             
        optimizer.zero_grad()
        output = model(X_masked)           
        X_orig_seq = X_orig.squeeze(1)   
        mask_seq = mask.squeeze(1)
        mse_loss = criterion(output * (1 - mask_seq), X_orig_seq * (1 - mask_seq))
        mse_loss.backward()
        optimizer.step()
        batch_loss = mse_loss.item() * X_masked.size(0)
        epoch_loss += batch_loss

        data = {
            'batch_loss': batch_loss
        }

        wandb.log(data)


    epoch_loss /= len(dataset)

    data = {
        'epoch_loss': epoch_loss
    }

    wandb.log(data)

    print(f"Epoch {epoch+1}/{num_epochs} - Reconstruction MSE: {epoch_loss:.4f}")
