"""
Visualize BERT reconstruction performance on the best features (43, 25, 22).
Cherry-picks flights with good reconstruction quality for presentation.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from torch.utils.data import DataLoader, TensorDataset
from ngafid_datasets.transformation_dataset import mask_transform
from tqdm import tqdm
from models.bert_masked_regressor import BertMaskedRegressor

# Feature indices and names for the top 3 features
BEST_FEATURES = {
    43: 'aoasimple',
    25: 'hplfd',
    22: 'fqtyr'
}

def load_bert_model(model_path, feat_dim, hidden_size, encoder_layers, decoder_layers, num_heads, max_seq_len, device):
    """Load trained BERT masked regressor model."""
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def evaluate_and_rank_flights(model, test_data, flight_ids, normalization_params, feature_idx,
                               batch_size=32, masking_ratio=0.5, mean_mask_length=60,
                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate all flights and rank by reconstruction quality for a specific feature.
    Returns flight indices sorted by MSE (best to worst).
    """
    model.eval()

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    test_dataset = TensorDataset(torch.FloatTensor(test_data_normalized), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    flight_mse_scores = []

    with torch.no_grad():
        batch_offset = 0
        for data, batch_ids in tqdm(test_loader, desc=f"Ranking flights for feature {feature_idx}", unit="batch"):
            data = data.to(device)

            original_data = data.cpu().numpy()
            masked_batch = []

            for sequence, flight_id in zip(original_data, batch_ids):
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

            reconstructed = model(masked_data)

            original_norm = data.cpu().numpy()
            recon_norm = reconstructed.cpu().numpy()

            # Compute MSE for this feature for each sample in batch
            for i in range(len(batch_ids)):
                mse = np.mean((original_norm[i, :, feature_idx] - recon_norm[i, :, feature_idx]) ** 2)
                flight_mse_scores.append((batch_offset + i, batch_ids[i].item(), mse))

            batch_offset += len(batch_ids)

    # Sort by MSE (ascending - best first)
    flight_mse_scores.sort(key=lambda x: x[2])

    return flight_mse_scores

def plot_reconstruction(original, reconstructed, mask, flight_id, feature_idx, feature_name,
                        output_path, mse_score):
    """
    Plot reconstruction for a single flight and feature with masked regions highlighted.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5

    original_color = '#2E86C1'
    reconstructed_color = '#E74C3C'
    mask_color = '#F7DC6F'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    time_steps = np.arange(len(original))

    # Full sequence plot
    ax1.plot(time_steps, original, alpha=0.8, linewidth=2.5,
             label='Ground Truth', color=original_color)
    ax1.plot(time_steps, reconstructed, alpha=0.8, linewidth=2.5,
             linestyle='--', label='BERT Reconstruction', color=reconstructed_color)

    # Highlight masked regions
    masked_regions = []
    in_mask = False
    start_idx = 0

    for i, m in enumerate(mask):
        if m and not in_mask:
            start_idx = i
            in_mask = True
        elif not m and in_mask:
            masked_regions.append((start_idx, i))
            in_mask = False

    if in_mask:
        masked_regions.append((start_idx, len(mask)))

    for start, end in masked_regions:
        ax1.axvspan(start, end, color=mask_color, alpha=0.3)

    # Add legend element for masked region
    ax1.axvspan(0, 0, color=mask_color, alpha=0.3, label='Masked Region')

    ax1.set_title(f'Full Sequence - Flight {flight_id} - Feature: {feature_name} (Index {feature_idx})\nMSE: {mse_score:.6f}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.5)

    # Zoomed view of a representative masked region
    if masked_regions:
        # Pick a masked region around the middle of the sequence
        mid_point = len(original) // 2
        closest_region = min(masked_regions, key=lambda x: abs((x[0] + x[1]) / 2 - mid_point))
        zoom_start, zoom_end = closest_region

        # Add padding around the masked region
        padding = max(20, (zoom_end - zoom_start) // 2)
        zoom_start = max(0, zoom_start - padding)
        zoom_end = min(len(original), zoom_end + padding)

        ax2.plot(time_steps[zoom_start:zoom_end], original[zoom_start:zoom_end],
                alpha=0.8, linewidth=2.5, label='Ground Truth', color=original_color)
        ax2.plot(time_steps[zoom_start:zoom_end], reconstructed[zoom_start:zoom_end],
                alpha=0.8, linewidth=2.5, linestyle='--', label='BERT Reconstruction',
                color=reconstructed_color)

        # Highlight the masked region in zoom
        for start, end in masked_regions:
            if start < zoom_end and end > zoom_start:
                ax2.axvspan(max(start, zoom_start), min(end, zoom_end),
                           color=mask_color, alpha=0.3)

        ax2.axvspan(0, 0, color=mask_color, alpha=0.3, label='Masked Region')

        ax2.set_title(f'Zoomed View - Flight {flight_id} - Feature: {feature_name}',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")

def visualize_best_flights(model, test_data, flight_ids, normalization_params,
                           num_flights_per_feature=3, batch_size=32,
                           masking_ratio=0.5, mean_mask_length=60,
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Cherry-pick and visualize the best reconstructed flights for each of the top 3 features.
    """

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    print("\n" + "=" * 80)
    print("CHERRY-PICKING BEST FLIGHTS FOR VISUALIZATION")
    print("=" * 80)

    for feature_idx, feature_name in BEST_FEATURES.items():
        print(f"\nProcessing feature {feature_idx} ({feature_name})...")

        # Rank all flights by reconstruction quality for this feature
        flight_rankings = evaluate_and_rank_flights(
            model, test_data, flight_ids, normalization_params, feature_idx,
            batch_size=batch_size, masking_ratio=masking_ratio,
            mean_mask_length=mean_mask_length, device=device
        )

        # Select the best flights
        best_flights = flight_rankings[:num_flights_per_feature]

        print(f"\nTop {num_flights_per_feature} flights for {feature_name}:")
        for rank, (idx, fid, mse) in enumerate(best_flights, 1):
            print(f"  Rank {rank}: Flight {fid} (index {idx}) - MSE: {mse:.6f}")

        # Visualize each selected flight
        print(f"\nGenerating visualizations for {feature_name}...")
        for rank, (idx, fid, mse) in enumerate(best_flights, 1):
            # Get the specific flight data
            flight_data_norm = test_data_normalized[idx]

            # Create mask
            _, masked_sequence, mask = mask_transform(
                flight_data_norm,
                masking_ratio=masking_ratio,
                mean_mask_length=mean_mask_length,
                mode='separate',
                distribution='geometric',
                random_seed=int(fid)
            )

            # Run reconstruction
            model.eval()
            with torch.no_grad():
                masked_input = torch.FloatTensor(masked_sequence.numpy()).unsqueeze(0).to(device)
                reconstructed = model(masked_input)
                reconstructed = reconstructed.cpu().numpy()[0]

            # Denormalize for visualization
            original_denorm = flight_data_norm * normalization_params['std'] + normalization_params['mean']
            recon_denorm = reconstructed * normalization_params['std'] + normalization_params['mean']

            # Plot
            output_path = f'best_reconstruction_feature_{feature_idx}_{feature_name}_flight_{fid}_rank_{rank}.png'
            plot_reconstruction(
                original_denorm[:, feature_idx],
                recon_denorm[:, feature_idx],
                mask.numpy()[:, feature_idx],
                fid,
                feature_idx,
                feature_name,
                output_path,
                mse
            )

    # Create summary CSV
    summary_data = []
    for feature_idx, feature_name in BEST_FEATURES.items():
        flight_rankings = evaluate_and_rank_flights(
            model, test_data, flight_ids, normalization_params, feature_idx,
            batch_size=batch_size, masking_ratio=masking_ratio,
            mean_mask_length=mean_mask_length, device=device
        )

        best_flights = flight_rankings[:num_flights_per_feature]
        for rank, (idx, fid, mse) in enumerate(best_flights, 1):
            summary_data.append({
                'feature_index': feature_idx,
                'feature_name': feature_name,
                'rank': rank,
                'flight_id': fid,
                'data_index': idx,
                'mse': mse
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('best_flights_summary.csv', index=False)
    print(f"\n✓ Summary saved to: best_flights_summary.csv")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize BERT reconstruction on best features with cherry-picked flights')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained BERT model weights')
    parser.add_argument('--norm_params_path', type=str, required=True,
                       help='Path to normalization parameters')
    parser.add_argument('--hidden_size', type=int, default=1536,
                       help='Hidden dimension size (default: 1536)')
    parser.add_argument('--encoder_layers', type=int, default=12,
                       help='Number of encoder layers (default: 12)')
    parser.add_argument('--decoder_layers', type=int, default=8,
                       help='Number of decoder layers (default: 8)')
    parser.add_argument('--num_heads', type=int, default=16,
                       help='Number of attention heads (default: 16)')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--masking_ratio', type=float, default=0.5,
                       help='Proportion of input to mask (default: 0.5)')
    parser.add_argument('--mean_mask_length', type=int, default=60,
                       help='Average length of masking subsequences (default: 60)')
    parser.add_argument('--num_flights', type=int, default=3,
                       help='Number of best flights to visualize per feature (default: 3)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading test data from {args.data_dir}...")
    from utils import load_flight_data
    test_data, flight_ids = load_flight_data(args.data_dir)
    feat_dim = test_data.shape[2]
    print(f"Loaded {len(flight_ids)} flights with {feat_dim} features")

    # Load normalization parameters
    print(f"Loading normalization parameters from {args.norm_params_path}...")
    normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()

    # Load model
    print(f"\nLoading BERT model from {args.model_path}...")
    model = load_bert_model(
        args.model_path,
        feat_dim=feat_dim,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        device=device
    )
    print("Model loaded successfully")

    # Visualize best flights
    visualize_best_flights(
        model,
        test_data,
        flight_ids,
        normalization_params,
        num_flights_per_feature=args.num_flights,
        batch_size=args.batch_size,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        device=device
    )
