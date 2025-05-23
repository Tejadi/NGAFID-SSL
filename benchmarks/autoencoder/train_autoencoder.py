import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from benchmarks.autoencoder.autoencoder import TimeSeriesAutoencoder
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
    print("Computing normalization parameters...")
    data_reshaped = train_data.reshape(-1, input_dim)
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    
    data_std[data_std == 0] = 1.0
    
    train_data_normalized = (train_data - data_mean) / data_std
    val_data_normalized = (val_data - data_mean) / data_std
    print("Normalization parameters computed.")
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data_normalized))
    val_dataset = TensorDataset(torch.FloatTensor(val_data_normalized))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
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
                print(masked_data.shape)
                reconstructed = model(masked_data)
                val_loss = criterion(reconstructed, data)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            })
    
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
    hidden_dim=16,
    batch_size=32,
    n_epochs=20,
    learning_rate=1e-4,
    mask_length=10,  
    start_point=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    wandb_run=None
):

    print("Computing normalization parameters...")
    data_reshaped = train_data.reshape(-1, input_dim)
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    

    data_std[data_std == 0] = 1.0
    

    train_data_normalized = (train_data - data_mean) / data_std
    val_data_normalized = (val_data - data_mean) / data_std
    print("Normalization parameters computed.")
    

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
    
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(n_epochs), desc='Training epochs'):
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
        
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            })
    
    normalization_params = {
        'mean': data_mean,
        'std': data_std
    }
    
    return model, normalization_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder on flight data')
    parser.add_argument('--train_data_dir', type=str, required=True,
                      help='Directory containing flight CSV files')
    parser.add_argument('--val_data_dir', type=str, required=True,
                      help='Directory containing validation flight CSV files')
    parser.add_argument('--sequence_length_csv', type=str,
                      help='Path to CSV file containing flight_id to sequence_length mapping (required for sequential masking)')
    parser.add_argument('--hidden_dim', type=int, default=16,
                      help='Hidden dimension size (default: 16)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=20,
                      help='Number of epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--mask_length', type=int,
                      help='Length of sequential mask (required for sequential masking)')
    parser.add_argument('--start_point', type=float,
                      help='Starting point for sequential mask as fraction of sequence length (required for sequential masking)')
    parser.add_argument('--masking_ratio', type=float,
                      help='Ratio of values to mask (required for random masking)')
    parser.add_argument('--mean_mask_length', type=int,
                      help='Mean length of random masks (required for random masking)')
    parser.add_argument('--job_name', type=str, required=True,
                      help='Name for the wandb run')
    parser.add_argument('--disable_wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    parser.add_argument('--use_sequential', action='store_true',
                      help='Use sequential masking instead of random masking')
    args = parser.parse_args()

    if args.use_sequential:
        if args.sequence_length_csv is None:
            parser.error("--sequence_length_csv is required when using sequential masking")
        if args.start_point is None:
            parser.error("--start_point is required when using sequential masking")
        if args.mask_length is None:
            parser.error("--mask_length is required when using sequential masking")
    else:
        if args.masking_ratio is None:
            parser.error("--masking_ratio is required when using random masking")
        if args.mean_mask_length is None:
            parser.error("--mean_mask_length is required when using random masking")

    train_data, train_ids = load_flight_data(args.train_data_dir)
    val_data, val_ids = load_flight_data(args.val_data_dir)
    sequence_length_map = load_sequence_lengths(args.sequence_length_csv)
    input_dim = train_data.shape[2]
    
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
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            wandb_run=wandb_run
        )
    
    model_name = "trained_sequential_autoencoder.pth" if args.use_sequential else "trained_autoencoder.pth"
    torch.save(trained_model.state_dict(), model_name)
    
    norm_params_name = "sequential_normalization_params.npy" if args.use_sequential else "normalization_params.npy"
    np.save(norm_params_name, normalization_params)
    
    if wandb_run is not None:
        wandb.finish() 