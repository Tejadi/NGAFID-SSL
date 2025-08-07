import os
import shutil

import torch
import yaml

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from benchmarks.autoencoder.autoencoder import TimeSeriesAutoencoder
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

def load_config(file: str = 'env.yml'):
    config_path = Path(__file__).parent / file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = Path(__file__).parent
    for key, path in config['paths'].items():
        config['paths'][key] = str(base_dir / path)
    
    return config

def load_flight_data(flight_dir):
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")
    
    flights = []
    flight_ids = []
    for path in tqdm(csv_files, desc='Loading flight data'):
        filename = path.name
        flight_id = int(filename.split('flight_')[1].split('.csv')[0])
        flight_ids.append(flight_id)
        
        flight = pd.read_csv(path)
        flight_array = flight.values
        flights.append(flight_array)
    
    flights_array = np.stack(flights, axis=0)
    return flights_array, flight_ids

def load_model(model_path, input_dim, hidden_dim, device):
    model = TimeSeriesAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def plot_reconstructions(original, reconstructed, feature_indices=[32, 33, 34], num_samples=10):
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5
    
    original_color = '#2E86C1'
    reconstructed_color = '#E74C3C'
    
    num_total_samples = original.shape[0]
    random_indices = np.random.choice(num_total_samples, num_samples, replace=False)
    
    for feature_idx in feature_indices:
        fig = plt.figure(figsize=(15, 5 * num_samples))
        for i, idx in enumerate(random_indices):
            plt.subplot(num_samples, 1, i + 1)
            
            time_steps = np.arange(original.shape[1])
            data_orig = {'Time Step': time_steps, 
                        'Value': original[idx, :, feature_idx],
                        'Type': ['Original'] * len(time_steps)}
            data_recon = {'Time Step': time_steps, 
                         'Value': reconstructed[idx, :, feature_idx],
                         'Type': ['Reconstructed'] * len(time_steps)}
            
            sns.lineplot(data=data_orig, x='Time Step', y='Value', label='Original', 
                        alpha=0.8, linewidth=2, color=original_color)
            sns.lineplot(data=data_recon, x='Time Step', y='Value', label='Reconstructed',
                        alpha=0.8, linewidth=2, color=reconstructed_color)
            
            plt.title(f'Sample {idx+1}, Feature {feature_idx}', pad=20, fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            plt.grid(True, color='#E5E5E5', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'reconstruction_comparison_feature_{feature_idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def get_aircraft_counts(data_dir):
    aircraft_counter, _ = count_aircraft_types(data_dir)
    return [count for _, count in sorted(aircraft_counter.items())]

def plot_aircraft_type_comparison(original, reconstructed, aircraft_type_counts=None, feature_indices=[34]):
    if aircraft_type_counts is None:
        aircraft_type_counts = [600, 297, 107]
        
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5
    
    original_color = '#2E86C1'
    reconstructed_color = '#E74C3C'
    
    start_indices = [0]
    for count in aircraft_type_counts[:-1]:
        start_indices.append(start_indices[-1] + count)
    
    selected_indices = []
    for i, count in enumerate(aircraft_type_counts):
        if count > 0:
            selected_indices.append(random.randint(start_indices[i], start_indices[i] + count - 1))
    
    print("\nAircraft types from left to right in plots:")
    for i in range(len(selected_indices)):
        print(f"Position {i+1}: Aircraft Type {i+1}")
    print()
    
    for feature_idx in feature_indices:
        fig = plt.figure(figsize=(20, 6))
        
        for i, idx in enumerate(selected_indices):
            ax = plt.subplot(1, len(selected_indices), i + 1)
            time_steps = np.arange(original.shape[1])
            
            plt.plot(time_steps, original[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, label='Original', color=original_color)
            
            plt.plot(time_steps, reconstructed[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, linestyle='--', label='Reconstructed', 
                    color=reconstructed_color)
            
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            
            ax.tick_params(axis='x', labelsize=10)
            plt.xticks(np.arange(0, len(time_steps), len(time_steps)//10))
            
            ax.grid(True, color='#E5E5E5', alpha=0.5)
        
        plt.tight_layout(w_pad=3.0)
        plt.savefig(f'aircraft_comparison_feature_{feature_idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

def load_sequence_lengths(csv_path):
    df = pd.read_csv(csv_path)
    sequence_length_map = dict(zip(df['flight_id'], df['length']))
    return sequence_length_map

def plot_sequential_reconstructions(original, reconstructed, flight_ids, feature_indices=[34], start_point=0.5, mask_length=10, sequence_length_csv=None, num_samples=5):
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5
    
    original_color = '#2E86C1'
    reconstructed_color = '#E74C3C'
    mask_color = '#F7DC6F'
    
    if sequence_length_csv:
        seq_lengths = pd.read_csv(sequence_length_csv)
        seq_length_map = dict(zip(seq_lengths['flight_id'], seq_lengths['length']))
    else:
        seq_length_map = {fid: original.shape[1] for fid in flight_ids}
    
    num_total_samples = original.shape[0]
    random_indices = np.random.choice(num_total_samples, min(num_samples, num_total_samples), replace=False)
    
    for feature_idx in feature_indices:
        for i, idx in enumerate(random_indices):
            flight_id = flight_ids[idx]
            seq_len = seq_length_map[flight_id]
            
            mask_start = int(start_point * seq_len)
            mask_end = min(mask_start + mask_length, seq_len)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            time_steps = np.arange(original.shape[1])
            
            ax1.plot(time_steps, original[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, label='Original', color=original_color)
            
            ax1.plot(time_steps, reconstructed[idx, :, feature_idx], 
                    alpha=0.8, linewidth=2, linestyle='--', label='Reconstructed', 
                    color=reconstructed_color)
            
            ax1.axvspan(mask_start, mask_end, color=mask_color, alpha=0.3, label='Masked Region')
            
            ax1.set_title(f'Full Sequence - Flight {flight_id}, Feature {feature_idx}')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Value')
            ax1.legend()
            
            padding = mask_length // 2
            zoom_start = max(0, mask_start - padding)
            zoom_end = min(seq_len, mask_end + padding)
            
            ax2.plot(time_steps[zoom_start:zoom_end], 
                    original[idx, zoom_start:zoom_end, feature_idx],
                    alpha=0.8, linewidth=2, label='Original', color=original_color)
            
            ax2.plot(time_steps[zoom_start:zoom_end], 
                    reconstructed[idx, zoom_start:zoom_end, feature_idx],
                    alpha=0.8, linewidth=2, linestyle='--', label='Reconstructed', 
                    color=reconstructed_color)
            
            ax2.axvspan(mask_start, mask_end, color=mask_color, alpha=0.3, label='Masked Region')
            
            ax2.set_title(f'Zoomed View of Masked Region - Flight {flight_id}, Feature {feature_idx}')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Value')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'sequential_reconstruction_flight_{flight_id}_feature_{feature_idx}.png',
                       bbox_inches='tight', dpi=300)
            plt.close()
