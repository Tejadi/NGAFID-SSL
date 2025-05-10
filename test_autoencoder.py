import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import TimeSeriesAutoencoder
from datasets.transformation_dataset import mask_transform
from train_autoencoder import load_flight_data
import argparse
from pathlib import Path
from tqdm import tqdm
import random
from count_aircraft_types import count_aircraft_types

def load_model(model_path, input_dim, hidden_dim, device):
    """
    Load a trained autoencoder model.
    """
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_data, flight_ids, normalization_params, batch_size=32, masking_ratio=0.6, mean_mask_length=3,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the autoencoder model on test data.
    
    Args:
        model: The trained autoencoder model
        test_data: Test data to evaluate on
        flight_ids: List of flight IDs to use as random seeds
        normalization_params: Dictionary containing 'mean' and 'std' for denormalization
        batch_size: Batch size for evaluation
        masking_ratio: Ratio of data to mask
        mean_mask_length: Mean length of masked sequences
        device: Device to run evaluation on
    """
    model.eval()
    
    # Normalize test data using saved parameters
    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']
    
    # Convert to PyTorch dataset and create a dataset that includes flight IDs
    test_dataset = TensorDataset(torch.FloatTensor(test_data_normalized), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    
    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            
            # Create masked version
            original_data = data.cpu().numpy()
            masked_batch = []
            for sequence, flight_id in zip(original_data, batch_ids):
                # Use flight_id as the random seed
                _, masked_sequence, mask = mask_transform(
                    sequence,
                    masking_ratio=masking_ratio,
                    mean_mask_length=mean_mask_length,
                    mode='separate',
                    distribution='geometric',
                    random_seed=int(flight_id)
                )
                masked_sequence = masked_sequence.numpy()
                masked_batch.append(masked_sequence)
            
            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)
            
            # Get reconstruction
            reconstructed = model(masked_data)
            
            # Denormalize both original and reconstructed data
            original_denorm = data.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            recon_denorm = reconstructed.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            
            # Calculate metrics on denormalized data
            mae = np.mean(np.abs(original_denorm - recon_denorm))
            mse = np.mean((original_denorm - recon_denorm) ** 2)
            
            total_mae += mae
            total_mse += mse
            num_batches += 1
            
            all_orig.append(original_denorm)
            all_recon.append(recon_denorm)
    
    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    rmse = np.sqrt(avg_mse)
    
    metrics = {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': rmse
    }
    
    return metrics, np.concatenate(all_orig), np.concatenate(all_recon)

def plot_reconstructions(original, reconstructed, feature_indices=[32, 33, 34], num_samples=10):
    """
    Plot original vs reconstructed sequences for visual comparison using seaborn.
    """
    # Set the seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_palette("husl")
    
    num_total_samples = original.shape[0]
    random_indices = np.random.choice(num_total_samples, num_samples, replace=False)
    
    for feature_idx in feature_indices:
        fig = plt.figure(figsize=(15, 5 * num_samples))
        for i, idx in enumerate(random_indices):
            plt.subplot(num_samples, 1, i + 1)
            
            # Create a DataFrame for better seaborn plotting
            time_steps = np.arange(original.shape[1])
            data_orig = {'Time Step': time_steps, 
                        'Value': original[idx, :, feature_idx],
                        'Type': ['Original'] * len(time_steps)}
            data_recon = {'Time Step': time_steps, 
                         'Value': reconstructed[idx, :, feature_idx],
                         'Type': ['Reconstructed'] * len(time_steps)}
            
            # Plot using seaborn
            sns.lineplot(data=data_orig, x='Time Step', y='Value', label='Original', 
                        alpha=0.7, linewidth=2)
            sns.lineplot(data=data_recon, x='Time Step', y='Value', label='Reconstructed',
                        alpha=0.7, linewidth=2)
            
            plt.title(f'Sample {idx+1}, Feature {feature_idx}', pad=20, fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f'reconstruction_comparison_feature_{feature_idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def get_aircraft_counts(data_dir):
    """
    Get the counts of each aircraft type from the data directory.
    Returns a list of counts in order of appearance.
    """
    aircraft_counter, _ = count_aircraft_types(data_dir)
    # Convert counter to list of counts in order of appearance
    return [count for _, count in sorted(aircraft_counter.items())]

def plot_aircraft_type_comparison(original, reconstructed, aircraft_type_counts=None, feature_indices=[32, 33, 34]):
    """
    Plot original vs reconstructed sequences for different aircraft types side by side.
    
    Args:
        original: Original data array (num_samples, timesteps, features)
        reconstructed: Reconstructed data array (num_samples, timesteps, features)
        aircraft_type_counts: List where each element represents count of each aircraft type in order.
                            If None, defaults to [600, 297, 107]
        feature_indices: List of feature indices to plot
    """
    if aircraft_type_counts is None:
        aircraft_type_counts = [600, 297, 107]
        
    # Set the seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_palette("husl")
    
    # Calculate starting indices for each aircraft type
    start_indices = [0]
    for count in aircraft_type_counts[:-1]:
        start_indices.append(start_indices[-1] + count)
    
    # For each aircraft type, randomly select one sample
    selected_indices = []
    for i, count in enumerate(aircraft_type_counts):
        if count > 0:
            selected_indices.append(random.randint(start_indices[i], start_indices[i] + count - 1))
    
    # Create plots for each feature
    for feature_idx in feature_indices:
        fig = plt.figure(figsize=(15, 10))
        
        # Plot original and reconstructed data for each aircraft type
        for i, idx in enumerate(selected_indices):
            # Original data
            plt.subplot(2, len(selected_indices), i + 1)
            time_steps = np.arange(original.shape[1])
            plt.plot(time_steps, original[idx, :, feature_idx], 
                    label='Original', alpha=0.7, linewidth=2)
            plt.plot(time_steps, reconstructed[idx, :, feature_idx], 
                    label='Reconstructed', alpha=0.7, linewidth=2, linestyle='--')
            plt.title(f'Aircraft Type {i+1}\nFeature {feature_idx} - Original vs Reconstructed')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            
            # Overlay comparison
            plt.subplot(2, 1, 2)
            plt.plot(time_steps, original[idx, :, feature_idx], 
                    label=f'Aircraft {i+1} Original', alpha=0.7, linewidth=2)
            plt.plot(time_steps, reconstructed[idx, :, feature_idx], 
                    label=f'Aircraft {i+1} Reconstructed', alpha=0.7, linewidth=2, linestyle='--')
        
        plt.title(f'Feature {feature_idx} - All Aircraft Types Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'aircraft_comparison_feature_{feature_idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained autoencoder on flight data')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--norm_params_path', type=str, required=True,
                      help='Path to normalization parameters')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--masking_ratio', type=float, default=0.6,
                      help='Proportion of input to mask (default: 0.6)')
    parser.add_argument('--mean_mask_length', type=int, default=3,
                      help='Average length of masking subsequences (default: 3)')
    
    args = parser.parse_args()
    
    # Get aircraft counts from the data directory
    print("Analyzing aircraft types in the data directory...")
    aircraft_counts = get_aircraft_counts(args.data_dir)
    print(f"Found aircraft counts: {aircraft_counts}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load normalization parameters
    normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()
    
    # Load test data
    test_data, flight_ids = load_flight_data(args.data_dir)
    input_dim = test_data.shape[2]
    
    # Load model
    model = load_model(args.model_path, input_dim, args.hidden_dim, device)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, orig_data, recon_data = evaluate_model(
        model,
        test_data,
        flight_ids,
        normalization_params=normalization_params,
        batch_size=args.batch_size,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        device=device
    )
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    
    # Plot reconstructions with aircraft type comparison
    print("\nGenerating aircraft type comparison plots...")
    plot_aircraft_type_comparison(orig_data, recon_data, aircraft_counts)
    print("Aircraft type comparison plots saved as 'aircraft_comparison_feature_X.png' for each feature")
    
    # Original reconstruction plots
    # print("\nGenerating general reconstruction plots...")
    # plot_reconstructions(orig_data, recon_data)
    # print("General reconstruction plots saved as 'reconstruction_comparison_feature_X.png' for each feature") 