#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from typing import Dict, Tuple

from models.patchtst_masked_regressor import PatchTSTMaskedRegressor
from ngafid_datasets.transformation_dataset import mask_transform
from utils import load_flight_data, get_aircraft_counts, plot_aircraft_type_comparison, plot_reconstructions


def load_patchtst_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    model = PatchTSTMaskedRegressor(
        feat_dim=args['feat_dim'],
        seq_len=args['seq_len'],
        patch_len=args.get('patch_len', 16),
        stride=args.get('stride', 8),
        d_model=args['d_model'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        encoder_layers=args['encoder_layers'],
        decoder_layers=args['decoder_layers'],
        dropout=args['dropout'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, args


def normalize_data_robust(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    normalized_data = np.copy(data)
    norm_params = {'medians': [], 'iqrs': []}

    for i in range(data.shape[1]):
        feature_data = data[:, i]

        median = np.median(feature_data)
        q75, q25 = np.percentile(feature_data, [75, 25])
        iqr = q75 - q25

        norm_params['medians'].append(median)
        norm_params['iqrs'].append(iqr)

        if iqr > 1e-6:
            normalized_data[:, i] = (feature_data - median) / iqr
        else:
            normalized_data[:, i] = feature_data - median

    return normalized_data, norm_params

def denormalize_data_robust(data: np.ndarray, norm_params: Dict) -> np.ndarray:
    denormalized_data = np.copy(data)

    for i in range(data.shape[1]):
        median = norm_params['medians'][i]
        iqr = norm_params['iqrs'][i]

        if iqr > 1e-6:
            denormalized_data[:, i] = data[:, i] * iqr + median
        else:
            denormalized_data[:, i] = data[:, i] + median

    return denormalized_data


def evaluate_patchtst_model(model, test_data, flight_ids,
                       masking_ratio=0.6, mean_mask_length=3, batch_size=16,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()

    test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    all_masks = []

    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating PatchTST model", unit="batch"):
            original_data = data.cpu().numpy()

            batch_orig_normalized = []
            batch_recon_denormalized = []
            batch_masks = []

            for sequence, flight_id in zip(original_data, batch_ids):
                normalized_seq, norm_params = normalize_data_robust(sequence)

                _, masked_sequence, mask = mask_transform(
                    normalized_seq,
                    masking_ratio=masking_ratio,
                    mean_mask_length=mean_mask_length,
                    mode='separate',
                    distribution='geometric',
                    random_seed=int(flight_id)
                )
                masked_sequence = masked_sequence.numpy()

                masked_input = torch.FloatTensor(masked_sequence).unsqueeze(0).to(device)
                reconstructed_normalized = model(masked_input).cpu().numpy().squeeze(0)

                reconstructed_original = denormalize_data_robust(reconstructed_normalized, norm_params)

                batch_orig_normalized.append(sequence)
                batch_recon_denormalized.append(reconstructed_original)
                batch_masks.append(mask.numpy())

            batch_orig = np.stack(batch_orig_normalized, axis=0)
            batch_recon = np.stack(batch_recon_denormalized, axis=0)

            mae = np.mean(np.abs(batch_orig - batch_recon))
            mse = np.mean((batch_orig - batch_recon) ** 2)

            total_mae += mae
            total_mse += mse
            num_batches += 1

            all_orig.append(batch_orig)
            all_recon.append(batch_recon)
            all_masks.append(np.stack(batch_masks, axis=0))

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
    parser = argparse.ArgumentParser(description='Test trained PatchTST masked regressor on flight data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PatchTST model checkpoint (.pt file)')
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, model_args = load_patchtst_model(args.model_path, device)

    print(f"Loading test data from {args.data_dir}")
    test_data, flight_ids = load_flight_data(args.data_dir)
    print(f"Loaded {len(flight_ids)} flight sequences with shape {test_data.shape}")

    print("Analyzing aircraft types in the data...")
    aircraft_counts = get_aircraft_counts(args.data_dir)
    print(f"Aircraft type counts: {aircraft_counts}")

    print("Evaluating PatchTST model with robust normalization...")
    print("Using per-sequence median + IQR normalization (matching training)")
    metrics, orig_data, recon_data, masks = evaluate_patchtst_model(
        model=model,
        test_data=test_data,
        flight_ids=flight_ids,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        batch_size=args.batch_size,
        device=device
    )

    print("\n" + "="*50)
    print("PATCHTST MASKED REGRESSOR TEST RESULTS")
    print("="*50)
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.6f}")
    print(f"MSE (Mean Squared Error):  {metrics['mse']:.6f}")
    print(f"RMSE (Root Mean Squared):  {metrics['rmse']:.6f}")
    print("="*50)

    results = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'metrics': metrics,
        'model_architecture': {
            'feat_dim': model_args['feat_dim'],
            'seq_len': model_args['seq_len'],
            'patch_len': model_args.get('patch_len', 16),
            'd_model': model_args['d_model'],
            'encoder_layers': model_args['encoder_layers'],
            'decoder_layers': model_args['decoder_layers'],
            'n_heads': model_args['n_heads'],
        },
        'evaluation_params': {
            'masking_ratio': args.masking_ratio,
            'mean_mask_length': args.mean_mask_length,
            'batch_size': args.batch_size,
        }
    }

    results_file = 'patchtst_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    print("\nGenerating visualization plots...")

    plot_aircraft_type_comparison(orig_data, recon_data, aircraft_counts, feature_indices=args.feature_indices)
    print(f"Aircraft comparison plots saved as 'aircraft_comparison_feature_X.png'")

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
