import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from autoencoder import TimeSeriesAutoencoder
import os
from pathlib import Path
import argparse
from datasets.transformation_dataset import mask_transform, sequential_mask_transform
from tqdm import tqdm
import wandb
from utils import load_flight_data, load_sequence_lengths

def train_autoencoder(
    train_data,
    val_data,
    input_dim,
    hidden_dim=64,
    batch_size=32,
    n_epochs=100,
    learning_rate=1e-3,
    masking_ratio=0.6,
    mean_mask_length=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    wandb_run=None
):
    # Compute normalization parameters using only training data
    print("Computing normalization parameters...")
    data_reshaped = train_data.reshape(-1, input_dim)
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    
    # Avoid division by zero in normalization
    data_std[data_std == 0] = 1.0
    
    # Normalize both training and validation data
    train_data_normalized = (train_data - data_mean) / data_std
    val_data_normalized = (val_data - data_mean) / data_std
    print("Normalization parameters computed.")
    
    # Convert data to PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(train_data_normalized))
    val_dataset = TensorDataset(torch.FloatTensor(val_data_normalized))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data,) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False)):
            data = data.to(device)
            
            original_data = data.cpu().numpy()
            masked_batch = []
            for sequence in original_data:
                _, masked_sequence, _ = mask_transform(
                    sequence,
                    masking_ratio=masking_ratio,
                    mean_mask_length=mean_mask_length,
                    mode='separate',
                    distribution='geometric'
                )
                masked_sequence = masked_sequence.numpy()
                masked_batch.append(masked_sequence)
            
            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)
            
            reconstructed = model(masked_data)
            loss = criterion(reconstructed, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0

        print("Validating...")
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to(device)
                
                original_data = data.cpu().numpy()
                masked_batch = []
                for sequence in original_data:
                    _, masked_sequence, _ = mask_transform(
                        sequence,
                        masking_ratio=masking_ratio,
                        mean_mask_length=mean_mask_length,
                        mode='separate',
                        distribution='geometric'
                    )
                    masked_sequence = masked_sequence.numpy()
                    masked_batch.append(masked_sequence)
                
                masked_data = np.stack(masked_batch, axis=0)
                masked_data = torch.FloatTensor(masked_data).to(device)
                
                reconstructed = model(masked_data)
                val_loss = criterion(reconstructed, data)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Log metrics to wandb if enabled
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            })
    
    # Save normalization parameters with the model
    normalization_params = {
        'mean': data_mean,
        'std': data_std
    }
    
    return model, normalization_params

def train_sequential_autoencoder(
    train_data,
    train_ids,
    val_data,
    val_ids,
    sequence_length_map,
    input_dim,
    hidden_dim=64,
    batch_size=32,
    n_epochs=100,
    learning_rate=1e-4,
    mask_length=10,  # Length of sequential mask
    start_point=0.5,  # Starting point for mask (as fraction of sequence length)
    device="cuda" if torch.cuda.is_available() else "cpu",
    wandb_run=None
):
    # Compute normalization parameters using only training data
    print("Computing normalization parameters...")
    data_reshaped = train_data.reshape(-1, input_dim)
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    
    # Avoid division by zero in normalization
    data_std[data_std == 0] = 1.0
    
    # Normalize both training and validation data
    train_data_normalized = (train_data - data_mean) / data_std
    val_data_normalized = (val_data - data_mean) / data_std
    print("Normalization parameters computed.")
    
    # Convert data to PyTorch datasets with flight IDs
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data_normalized),
        torch.LongTensor([sequence_length_map[id] for id in train_ids])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data_normalized),
        torch.LongTensor([sequence_length_map[id] for id in val_ids])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, seq_lengths) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False)):
            data = data.to(device)
            
            original_data = data.cpu().numpy()
            masked_batch = []
            for sequence, seq_len in zip(original_data, seq_lengths):
                _, masked_sequence, _ = sequential_mask_transform(
                    sequence,
                    starting_point=start_point,
                    n=mask_length,
                    sequence_length=seq_len.item()
                )
                masked_sequence = masked_sequence.numpy()
                masked_batch.append(masked_sequence)
            
            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)
            
            reconstructed = model(masked_data)
            loss = criterion(reconstructed, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0

        print("Validating...")
        with torch.no_grad():
            for batch_idx, (data, seq_lengths) in enumerate(val_loader):
                data = data.to(device)
                
                original_data = data.cpu().numpy()
                masked_batch = []
                for sequence, seq_len in zip(original_data, seq_lengths):
                    _, masked_sequence, _ = sequential_mask_transform(
                        sequence,
                        starting_point=start_point,
                        n=mask_length,
                        sequence_length=seq_len.item()
                    )
                    masked_sequence = masked_sequence.numpy()
                    masked_batch.append(masked_sequence)
                
                masked_data = np.stack(masked_batch, axis=0)
                masked_data = torch.FloatTensor(masked_data).to(device)
                
                reconstructed = model(masked_data)
                val_loss = criterion(reconstructed, data)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Log metrics to wandb if enabled
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            })
    
    # Save normalization parameters with the model
    normalization_params = {
        'mean': data_mean,
        'std': data_std
    }
    
    return model, normalization_params

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train autoencoder on flight data')
    parser.add_argument('--train_data_dir', type=str, required=True,
                      help='Directory containing flight CSV files')
    parser.add_argument('--val_data_dir', type=str, required=True,
                      help='Directory containing validation flight CSV files')
    parser.add_argument('--sequence_length_csv', type=str, required=True,
                      help='Path to CSV file containing flight_id to sequence_length mapping')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=100,
                      help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--mask_length', type=int, default=10,
                      help='Length of sequential mask (default: 10)')
    parser.add_argument('--start_point', type=float, default=0.5,
                      help='Starting point for sequential mask as fraction of sequence length (default: 0.5)')
    parser.add_argument('--job_name', type=str, required=True,
                      help='Name for the wandb run')
    parser.add_argument('--disable_wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    parser.add_argument('--use_sequential', action='store_true',
                      help='Use sequential masking instead of random masking')
    args = parser.parse_args()

    # Load and prepare data
    train_data, train_ids = load_flight_data(args.train_data_dir)
    val_data, val_ids = load_flight_data(args.val_data_dir)
    sequence_length_map = load_sequence_lengths(args.sequence_length_csv)
    input_dim = train_data.shape[2]
    
    # Initialize wandb if not disabled
    if not args.disable_wandb:
        wandb.init(
            project="ngafid-ssl-fall-24",
            entity="ngafid-ssl",
            name=args.job_name,
            config={
                'learning_rate': args.learning_rate,
                'epochs': args.n_epochs,
                'batch_size': args.batch_size,
                'hidden_dim': args.hidden_dim,
                'mask_length': args.mask_length if args.use_sequential else None,
                'start_point': args.start_point if args.use_sequential else None,
                'masking_type': 'sequential' if args.use_sequential else 'random',
                'input_dim': input_dim,
                'architecture': 'TimeSeriesAutoencoder',
                'device': "cuda" if torch.cuda.is_available() else "cpu",
                'train_data_size': len(train_data),
                'val_data_size': len(val_data),
                'num_train_flights': len(train_ids),
                'num_val_flights': len(val_ids)
            }
        )
        wandb_run = wandb
    else:
        wandb_run = None
    
    # Train the model
    if args.use_sequential:
        trained_model, normalization_params = train_sequential_autoencoder(
            train_data=train_data,
            train_ids=train_ids,
            val_data=val_data,
            val_ids=val_ids,
            sequence_length_map=sequence_length_map,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            mask_length=args.mask_length,
            start_point=args.start_point,
            wandb_run=wandb_run
        )
    else:
        trained_model, normalization_params = train_autoencoder(
            train_data=train_data,
            val_data=val_data,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            masking_ratio=0.6,
            mean_mask_length=3,
            wandb_run=wandb_run
        )
    
    # Save the trained model
    model_name = "trained_sequential_autoencoder.pth" if args.use_sequential else "trained_autoencoder.pth"
    torch.save(trained_model.state_dict(), model_name)
    
    # Save normalization parameters
    norm_params_name = "sequential_normalization_params.npy" if args.use_sequential else "normalization_params.npy"
    np.save(norm_params_name, normalization_params)
    
    # Close wandb run if it was used
    if wandb_run is not None:
        wandb.finish() 