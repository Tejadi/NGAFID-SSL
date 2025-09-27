#!/usr/bin/env python3
"""
Test script for BERT masked regressor on flight data.

Loads trained BERT model and evaluates it on flight data using proper loss normalization
similar to the autoencoder test script. Computes normalized MAE, MSE, and RMSE metrics
suitable for research papers.
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from typing import Dict, Tuple

# Import model and dataset utilities
from models.bert_masked_regressor import BertMaskedRegressor
from ngafid_datasets.transformation_dataset import mask_transform
from utils import load_flight_data, get_aircraft_counts, plot_aircraft_type_comparison, plot_reconstructions


class SimpleFlightDecoder(torch.nn.Module):
    """Simple decoder compatible with old model checkpoints."""

    def __init__(self, hidden_size: int, feat_dim: int, decoder_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        # Simple sequential decoder matching old architecture
        layers = []
        current_dim = hidden_size

        for i in range(decoder_layers):
            if i == decoder_layers - 1:
                # Final layer
                layers.append(torch.nn.Linear(current_dim, feat_dim))
            else:
                # Intermediate layers - reduce dimension gradually
                if i == 0:
                    next_dim = hidden_size // 2
                else:
                    next_dim = current_dim // 2

                layers.extend([
                    torch.nn.Linear(current_dim, next_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ])
                current_dim = next_dim

        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


def load_bert_model(checkpoint_path, device):
    """Load trained BERT model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    # Check if this is an old model format by looking at state dict keys
    model_keys = list(checkpoint['model_state_dict'].keys())
    is_old_format = any('decoder.decoder.' in key for key in model_keys)

    if is_old_format:
        print("Detected old model format, using compatible decoder")
        # Create model with simple decoder for compatibility
        from models.bert_masked_regressor import FlightBertEncoder

        encoder = FlightBertEncoder(
            feat_dim=args['feat_dim'],
            hidden_size=args['hidden_size'],
            num_layers=args['encoder_layers'],
            num_heads=args['num_heads'],
            dropout=args['dropout'],
            max_position_embeddings=args['seq_len'],
        )

        decoder = SimpleFlightDecoder(
            hidden_size=args['hidden_size'],
            feat_dim=args['feat_dim'],
            decoder_layers=args['decoder_layers'],
            dropout=args['dropout'],
        )

        class CompatibleBertMaskedRegressor(torch.nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.feat_dim = args['feat_dim']
                self.hidden_size = args['hidden_size']

            def forward(self, x_masked, attention_mask=None):
                encoded = self.encoder(x_masked, attention_mask)
                reconstructed = self.decoder(encoded)
                return reconstructed

        model = CompatibleBertMaskedRegressor(encoder, decoder).to(device)
    else:
        # Create model with new architecture
        model = BertMaskedRegressor(
            feat_dim=args['feat_dim'],
            hidden_size=args['hidden_size'],
            encoder_layers=args['encoder_layers'],
            decoder_layers=args['decoder_layers'],
            num_heads=args['num_heads'],
            dropout=args['dropout'],
            max_seq_len=args['seq_len'],
        ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, args


def normalize_data_robust(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Apply robust normalization using median and IQR, matching BERT training.

    Args:
        data: Flight data array of shape (seq_len, feat_dim)

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    normalized_data = np.copy(data)
    norm_params = {'medians': [], 'iqrs': []}

    for i in range(data.shape[1]):  # For each feature
        feature_data = data[:, i]

        # Use median and IQR for robust normalization (matching training)
        median = np.median(feature_data)
        q75, q25 = np.percentile(feature_data, [75, 25])
        iqr = q75 - q25

        norm_params['medians'].append(median)
        norm_params['iqrs'].append(iqr)

        # Avoid division by zero
        if iqr > 1e-6:
            normalized_data[:, i] = (feature_data - median) / iqr
        else:
            # If no variation, center around median
            normalized_data[:, i] = feature_data - median

    return normalized_data, norm_params

def denormalize_data_robust(data: np.ndarray, norm_params: Dict) -> np.ndarray:
    """
    Denormalize data using stored median and IQR parameters.

    Args:
        data: Normalized data array
        norm_params: Dictionary with 'medians' and 'iqrs' lists

    Returns:
        Denormalized data array
    """
    denormalized_data = np.copy(data)

    for i in range(data.shape[1]):
        median = norm_params['medians'][i]
        iqr = norm_params['iqrs'][i]

        if iqr > 1e-6:
            denormalized_data[:, i] = data[:, i] * iqr + median
        else:
            denormalized_data[:, i] = data[:, i] + median

    return denormalized_data


def evaluate_bert_model(model, test_data, flight_ids,
                       masking_ratio=0.6, mean_mask_length=3, batch_size=16,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate BERT model on test data using robust normalization matching training.

    Uses per-sequence robust normalization (median + IQR) just like during training.
    Returns metrics comparable to research papers by denormalizing before computing loss.
    """
    model.eval()

    # Create dataset with original (unnormalized) data
    test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    all_masks = []

    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating BERT model", unit="batch"):
            # Work with original data for this batch
            original_data = data.cpu().numpy()

            # Process each sequence individually (as in training)
            batch_orig_normalized = []
            batch_recon_denormalized = []
            batch_masks = []

            for sequence, flight_id in zip(original_data, batch_ids):
                # Step 1: Apply robust normalization to this sequence (matching training)
                normalized_seq, norm_params = normalize_data_robust(sequence)

                # Step 2: Apply masking to normalized sequence
                _, masked_sequence, mask = mask_transform(
                    normalized_seq,
                    masking_ratio=masking_ratio,
                    mean_mask_length=mean_mask_length,
                    mode='separate',
                    distribution='geometric',
                    random_seed=int(flight_id)
                )
                masked_sequence = masked_sequence.numpy()

                # Step 3: Forward pass through BERT model
                masked_input = torch.FloatTensor(masked_sequence).unsqueeze(0).to(device)
                reconstructed_normalized = model(masked_input).cpu().numpy().squeeze(0)

                # Step 4: Denormalize reconstructed data back to original scale
                reconstructed_original = denormalize_data_robust(reconstructed_normalized, norm_params)

                # Store results
                batch_orig_normalized.append(sequence)  # Keep original for loss computation
                batch_recon_denormalized.append(reconstructed_original)
                batch_masks.append(mask.numpy())

            # Convert to arrays
            batch_orig = np.stack(batch_orig_normalized, axis=0)
            batch_recon = np.stack(batch_recon_denormalized, axis=0)

            # Compute metrics on original scale
            mae = np.mean(np.abs(batch_orig - batch_recon))
            mse = np.mean((batch_orig - batch_recon) ** 2)

            total_mae += mae
            total_mse += mse
            num_batches += 1

            # Store for visualization
            all_orig.append(batch_orig)
            all_recon.append(batch_recon)
            all_masks.append(np.stack(batch_masks, axis=0))

    # Compute final metrics
    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    rmse = np.sqrt(avg_mse)

    metrics = {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': rmse
    }

    return metrics, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_masks)


def main():
    parser = argparse.ArgumentParser(description='Test trained BERT masked regressor on flight data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained BERT model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--masking_ratio', type=float, default=0.6,
                       help='Masking ratio for evaluation (default: 0.6)')
    parser.add_argument('--mean_mask_length', type=int, default=3,
                       help='Mean mask length for evaluation (default: 3)')
    parser.add_argument('--feature_indices', type=int, nargs='+', default=[34],
                       help='Feature indices to visualize (default: [34])')
    parser.add_argument('--num_visualization_samples', type=int, default=5,
                       help='Number of samples to visualize (default: 5)')

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model, model_args = load_bert_model(args.model_path, device)

    # Load test data
    print(f"Loading test data from {args.data_dir}")
    test_data, flight_ids = load_flight_data(args.data_dir)
    print(f"Loaded {len(flight_ids)} flight sequences with shape {test_data.shape}")

    # Analyze aircraft types
    print("Analyzing aircraft types in the data...")
    aircraft_counts = get_aircraft_counts(args.data_dir)
    print(f"Aircraft type counts: {aircraft_counts}")

    # Evaluate model using robust normalization (matching training)
    print("Evaluating BERT model with robust normalization...")
    print("Using per-sequence median + IQR normalization (matching training)")
    metrics, orig_data, recon_data, masks = evaluate_bert_model(
        model=model,
        test_data=test_data,
        flight_ids=flight_ids,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        batch_size=args.batch_size,
        device=device
    )

    # Print results
    print("\n" + "="*50)
    print("BERT MASKED REGRESSOR TEST RESULTS")
    print("="*50)
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.6f}")
    print(f"MSE (Mean Squared Error):  {metrics['mse']:.6f}")
    print(f"RMSE (Root Mean Squared):  {metrics['rmse']:.6f}")
    print("="*50)

    # Save results
    results = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'metrics': metrics,
        'model_architecture': {
            'feat_dim': model_args['feat_dim'],
            'hidden_size': model_args['hidden_size'],
            'encoder_layers': model_args['encoder_layers'],
            'decoder_layers': model_args['decoder_layers'],
            'num_heads': model_args['num_heads'],
        },
        'evaluation_params': {
            'masking_ratio': args.masking_ratio,
            'mean_mask_length': args.mean_mask_length,
            'batch_size': args.batch_size,
        }
    }

    results_file = 'bert_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Generate visualizations
    print("\nGenerating visualization plots...")

    # Aircraft type comparison plots
    plot_aircraft_type_comparison(orig_data, recon_data, aircraft_counts, feature_indices=args.feature_indices)
    print(f"Aircraft comparison plots saved as 'aircraft_comparison_feature_X.png'")

    # Reconstruction comparison plots
    plot_reconstructions(
        orig_data,
        recon_data,
        feature_indices=args.feature_indices,
        num_samples=args.num_visualization_samples
    )
    print(f"Reconstruction plots saved as 'reconstruction_comparison_feature_X.png'")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()