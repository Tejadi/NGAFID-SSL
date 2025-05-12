import os
import shutil

import torch
import yaml

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from autoencoder import TimeSeriesAutoencoder
from count_aircraft_types import count_aircraft_types
import matplotlib.pyplot as plt
import seaborn as sns
import random

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_config():
    config_path = Path(__file__).parent / 'new_env.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    # Make sure load_config works without converting paths to absolute first
    
    base_dir = Path(__file__).parent
    for key, path in config['paths'].items():
        config['paths'][key] = str(base_dir / path)
    
    return config

def load_flight_data(flight_dir):
    # Get all CSV files in the directory
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")
    
    flights = []
    flight_ids = []
    for path in tqdm(csv_files, desc='Loading flight data'):
        # Extract flight ID from filename
        filename = path.name
        # Find the number between 'flight_' and '.csv'
        flight_id = int(filename.split('flight_')[1].split('.csv')[0])
        flight_ids.append(flight_id)
        
        # Read CSV
        flight = pd.read_csv(path)
        # Convert to numpy array
        flight_array = flight.values
        flights.append(flight_array)
    
    # Stack all flights into a single array
    # This will give you (N, T, F) shape
    flights_array = np.stack(flights, axis=0)
    return flights_array, flight_ids

def load_model(model_path, input_dim, hidden_dim, device):
    """
    Load a trained autoencoder model.
    """
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def plot_reconstructions(original, reconstructed, feature_indices=[32, 33, 34], num_samples=10):
    """
    Plot original vs reconstructed sequences for visual comparison using seaborn.
    """
    # Set the seaborn style with improved grid settings
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Define contrasting colors
    original_color = '#2E86C1'  # Strong blue
    reconstructed_color = '#E74C3C'  # Strong red
    
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
            
            # Plot using seaborn with specific colors
            sns.lineplot(data=data_orig, x='Time Step', y='Value', label='Original', 
                        alpha=0.8, linewidth=2, color=original_color)
            sns.lineplot(data=data_recon, x='Time Step', y='Value', label='Reconstructed',
                        alpha=0.8, linewidth=2, color=reconstructed_color)
            
            plt.title(f'Sample {idx+1}, Feature {feature_idx}', pad=20, fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            # Ensure grid is visible with custom settings
            plt.grid(True, color='#E5E5E5', alpha=0.5)
        
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

def plot_aircraft_type_comparison(orig_data, recon_data, aircraft_counts, masks=None):
    """
    Plot comparison of original and reconstructed data for each feature, with data points colored by aircraft type.
    Also shows masked regions if masks are provided.
    
    Args:
        orig_data: Original data array
        recon_data: Reconstructed data array
        aircraft_counts: Dictionary mapping aircraft types to their counts
        masks: Boolean array indicating which regions were masked (True = kept, False = masked)
    """
    n_features = orig_data.shape[2]
    
    for feature_idx in range(n_features):
        plt.figure(figsize=(15, 6))
        
        # Plot original vs reconstructed for all points
        plt.subplot(1, 2, 1)
        plt.scatter(orig_data[:, :, feature_idx].flatten(), 
                   recon_data[:, :, feature_idx].flatten(),
                   alpha=0.1, s=1)
        
        # Add diagonal line
        min_val = min(orig_data[:, :, feature_idx].min(), recon_data[:, :, feature_idx].min())
        max_val = max(orig_data[:, :, feature_idx].max(), recon_data[:, :, feature_idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.xlabel('Original Values')
        plt.ylabel('Reconstructed Values')
        plt.title(f'Feature {feature_idx} Reconstruction')
        
        # Plot time series comparison for a random flight
        plt.subplot(1, 2, 2)
        random_flight = np.random.randint(0, orig_data.shape[0])
        
        # Plot original data
        plt.plot(orig_data[random_flight, :, feature_idx], 
                label='Original', alpha=0.7)
        
        # Plot reconstructed data
        plt.plot(recon_data[random_flight, :, feature_idx], 
                label='Reconstructed', alpha=0.7)
        
        # If masks are provided, show masked regions
        if masks is not None:
            mask = masks[random_flight, :, feature_idx]
            # Create shaded regions for masked areas
            for i in range(len(mask)):
                if not mask[i]:  # If the point was masked
                    plt.axvspan(i-0.5, i+0.5, color='gray', alpha=0.2)
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Feature {feature_idx} Time Series (Random Flight)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'aircraft_comparison_feature_{feature_idx}.png')
        plt.close()

def load_sequence_lengths(csv_path):
    """
    Load sequence lengths from a CSV file and create a mapping dictionary.
    
    Args:
        csv_path (str): Path to CSV file containing flight_id to sequence_length mapping
        
    Returns:
        dict: Dictionary mapping flight_id to sequence_length
    """
    df = pd.read_csv(csv_path)
    # Assuming CSV has columns 'flight_id' and 'sequence_length'
    sequence_length_map = dict(zip(df['flight_id'], df['length']))
    return sequence_length_map