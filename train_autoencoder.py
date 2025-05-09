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
from datasets.transformation_dataset import mask_transform
from tqdm import tqdm
import wandb

def load_flight_data(flight_dir):
    # Get all CSV files in the directory
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")
    
    flights = []
    for path in tqdm(csv_files, desc='Loading flight data'):
        # Read CSV
        flight = pd.read_csv(path)
        # Convert to numpy array
        flight_array = flight.values
        flights.append(flight_array)
    
    # Stack all flights into a single array
    # This will give you (N, T, F) shape
    flights_array = np.stack(flights, axis=0)
    return flights_array

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
                _, masked_sequence = mask_transform(
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
                    _, masked_sequence = mask_transform(
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train autoencoder on flight data')
    parser.add_argument('--train_data_dir', type=str, required=True,
                      help='Directory containing flight CSV files')
    parser.add_argument('--val_data_dir', type=str, required=True,
                      help='Directory containing validation flight CSV files')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=100,
                      help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--masking_ratio', type=float, default=0.6,
                      help='Proportion of input to mask (default: 0.6)')
    parser.add_argument('--mean_mask_length', type=int, default=7,
                      help='Average length of masking subsequences (default: 3)')
    parser.add_argument('--job_name', type=str, required=True,
                      help='Name for the wandb run')
    parser.add_argument('--disable_wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    args = parser.parse_args()

    # Load and prepare data
    train_data = load_flight_data(args.train_data_dir)
    val_data = load_flight_data(args.val_data_dir)
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
                'masking_ratio': args.masking_ratio,
                'mean_mask_length': args.mean_mask_length,
                'input_dim': input_dim,
                'architecture': 'TimeSeriesAutoencoder',
                'device': "cuda" if torch.cuda.is_available() else "cpu",
                'train_data_size': len(train_data),
                'val_data_size': len(val_data)
            }
        )
        wandb_run = wandb
    else:
        wandb_run = None
    
    # Train the model
    trained_model, normalization_params = train_autoencoder(
        train_data=train_data,
        val_data=val_data,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        wandb_run=wandb_run
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "trained_autoencoder.pth")
    
    # Save normalization parameters
    np.save("normalization_params.npy", normalization_params)
    
    # Close wandb run if it was used
    if wandb_run is not None:
        wandb.finish() 