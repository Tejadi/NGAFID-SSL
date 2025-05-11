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

def plot_aircraft_type_comparison(original, reconstructed, aircraft_type_counts=None, feature_indices=[34]):
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
        
    # Set the style with improved grid settings
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Define contrasting colors
    original_color = '#2E86C1'  # Strong blue
    reconstructed_color = '#E74C3C'  # Strong red
    
    # Calculate starting indices for each aircraft type
    start_indices = [0]
    for count in aircraft_type_counts[:-1]:
        start_indices.append(start_indices[-1] + count)
    
    # For each aircraft type, randomly select one sample
    selected_indices = []
    for i, count in enumerate(aircraft_type_counts):
        if count > 0:
            selected_indices.append(random.randint(start_indices[i], start_indices[i] + count - 1))
    
    # Print aircraft types in order
    print("\nAircraft types from left to right in plots:")
    for i in range(len(selected_indices)):
        print(f"Position {i+1}: Aircraft Type {i+1}")
    print()
    
    # Create plots for each feature
    for feature_idx in feature_indices:
        # Make the figure wider and slightly taller
        fig = plt.figure(figsize=(20, 6))
        
        # Plot original and reconstructed data for each aircraft type
        for i, idx in enumerate(selected_indices):
            ax = plt.subplot(1, len(selected_indices), i + 1)
            time_steps = np.arange(original.shape[1])
            
            # Plot original data with specific color
            plt.plot(time_steps, original[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, label='Original', color=original_color)
            
            # Plot reconstructed data with specific color
            plt.plot(time_steps, reconstructed[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, linestyle='--', label='Reconstructed', 
                    color=reconstructed_color)
            
            plt.xlabel('Time Step')
            plt.ylabel('Pitch Angle (deg)')
            plt.legend()
            
            # Improve x-axis readability
            ax.tick_params(axis='x', labelsize=10)
            # Add more x-axis ticks
            plt.xticks(np.arange(0, len(time_steps), len(time_steps)//10))
            
            # Ensure grid is visible with custom settings
            ax.grid(True, color='#E5E5E5', alpha=0.5)
        
        # Adjust layout with more width between subplots
        plt.tight_layout(w_pad=3.0)
        plt.savefig(f'aircraft_comparison_feature_{feature_idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()