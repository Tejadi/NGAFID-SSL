import torch
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ngafid_datasets.transformation_dataset import mask_transform, sequential_mask_transform
from utils import load_sequence_lengths
from models.bert_masked_regressor import BertMaskedRegressor
from tqdm import tqdm


def load_flight_data_with_filenames(flight_dir):
    """Load flight data and return filenames along with IDs."""
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")

    flights = []
    flight_ids = []
    flight_filenames = []
    for path in tqdm(csv_files, desc='Loading flight data'):
        filename = path.name
        flight_id = int(filename.split('flight_')[1].split('.csv')[0])
        flight_ids.append(flight_id)
        flight_filenames.append(filename)

        flight = pd.read_csv(path)
        flight_array = flight.values
        flights.append(flight_array)

    flights_array = np.stack(flights, axis=0)
    return flights_array, flight_ids, flight_filenames


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


def get_feature_names(data_dir):
    """Extract feature names from the first CSV file in the data directory."""
    # Get list of CSV files directly from the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and
                 not any(name in f.lower() for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None

    csv_path = os.path.join(data_dir, csv_files[0])
    print(f"Extracting feature names from: {csv_path}")

    try:
        # Read just the header
        df = pd.read_csv(csv_path, nrows=0)
        feature_names = df.columns.tolist()
        print(f"Successfully extracted {len(feature_names)} feature names")
        return feature_names
    except Exception as e:
        print(f"Warning: Could not extract feature names: {e}")
        return None


def visualize_single_flight(model, test_data, flight_ids, flight_filenames, feature_idx, normalization_params,
                           masking_ratio=0.5, mean_mask_length=60, use_sequential=False,
                           mask_length=10, start_point=0.5, sequence_length_map=None,
                           aircraft_name=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize reconstruction for a single flight (random or by aircraft name).
    """
    model.eval()

    # Select flight based on aircraft name or randomly
    if aircraft_name:
        # Search for flights matching the aircraft name
        matching_indices = [i for i, fname in enumerate(flight_filenames) if aircraft_name in fname]

        if not matching_indices:
            print(f"Warning: No flights found matching '{aircraft_name}'")
            print(f"Available aircraft types in dataset:")
            # Extract unique aircraft types from filenames
            aircraft_types = set()
            for fname in flight_filenames:
                # Extract aircraft name (everything before "_flight_")
                parts = fname.split('_flight_')
                if len(parts) > 0:
                    aircraft_types.add(parts[0])
            for atype in sorted(aircraft_types):
                print(f"  - {atype}")
            print("\nFalling back to random flight selection...")
            random_idx = np.random.randint(0, len(flight_ids))
        else:
            # Randomly select from matching flights
            random_idx = np.random.choice(matching_indices)
            print(f"Found {len(matching_indices)} flights matching '{aircraft_name}'")
    else:
        # Select random flight
        random_idx = np.random.randint(0, len(flight_ids))

    flight_id = flight_ids[random_idx]
    flight_filename = flight_filenames[random_idx]
    flight_data = test_data[random_idx]

    print(f"Selected Flight: {flight_filename}")
    print(f"Flight ID: {flight_id}")

    # Normalize
    flight_normalized = (flight_data - normalization_params['mean']) / normalization_params['std']

    # Create mask
    if use_sequential:
        seq_len = sequence_length_map[flight_id] if sequence_length_map else flight_data.shape[0]
        _, masked_sequence, mask = sequential_mask_transform(
            flight_normalized,
            starting_point=start_point,
            n=mask_length,
            sequence_length=seq_len
        )
    else:
        _, masked_sequence, mask = mask_transform(
            flight_normalized,
            masking_ratio=masking_ratio,
            mean_mask_length=mean_mask_length,
            mode='separate',
            distribution='geometric',
            random_seed=int(flight_id)
        )

    # Convert to tensor and add batch dimension
    masked_sequence = torch.FloatTensor(masked_sequence.numpy()).unsqueeze(0).to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructed = model(masked_sequence)

    # Denormalize
    original_denorm = flight_data
    recon_denorm = reconstructed.cpu().numpy()[0] * normalization_params['std'] + normalization_params['mean']

    return flight_id, flight_filename, original_denorm, recon_denorm, mask.numpy()


def plot_three_flights_side_by_side(flights_data, feature_idx, feature_name=None, masking_ratio=None, mean_mask_length=None):
    """
    Plot reconstruction for three flights side by side and save as PDF.

    flights_data: list of tuples (flight_id, flight_filename, original, reconstructed, aircraft_name)
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.alpha'] = 0.5

    original_color = '#2E86C1'
    reconstructed_color = '#E74C3C'

    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for idx, (flight_id, flight_filename, original, reconstructed, aircraft_name) in enumerate(flights_data):
        ax = axes[idx]
        time_steps = np.arange(original.shape[0])

        # Plot original and reconstructed
        ax.plot(time_steps, original[:, feature_idx],
                alpha=0.8, linewidth=2, label='Original', color=original_color)
        ax.plot(time_steps, reconstructed[:, feature_idx],
                alpha=0.8, linewidth=2, linestyle='--', label='Reconstructed',
                color=reconstructed_color)

        # Extract just the aircraft type from filename
        aircraft_type = flight_filename.split('_flight_')[0]
        ax.set_title(aircraft_type, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Pitch Angle (deg)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)

    plt.tight_layout()

    # Build output filename with feature name and masking parameters
    filename_parts = [feature_name if feature_name else f"feature_{feature_idx}"]
    if masking_ratio is not None:
        filename_parts.append(f"ratio{masking_ratio}")
    if mean_mask_length is not None:
        filename_parts.append(f"len{mean_mask_length}")

    output_filename = f'three_aircraft_{"_".join(filename_parts)}.pdf'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300, format='pdf')
    plt.close()

    print(f"✓ Visualization saved as: {output_filename}")
    return output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize reconstruction of a feature for a single random flight')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--norm_params_path', type=str, required=True,
                      help='Path to normalization parameters')
    parser.add_argument('--feature', type=str, required=True,
                      help='Feature name to visualize (e.g., magvar, e1oilp, baroa)')
    parser.add_argument('--aircraft_names', type=str, nargs=3, required=True,
                      help='Three aircraft names to compare side by side (e.g., "Cessna_172S" "PA-28-181" "PA-44-180")')
    parser.add_argument('--hidden_size', type=int, default=1024,
                      help='Hidden dimension size (default: 1024)')
    parser.add_argument('--encoder_layers', type=int, default=8,
                      help='Number of encoder layers (default: 8)')
    parser.add_argument('--decoder_layers', type=int, default=6,
                      help='Number of decoder layers (default: 6)')
    parser.add_argument('--num_heads', type=int, default=16,
                      help='Number of attention heads (default: 16)')
    parser.add_argument('--max_seq_len', type=int, default=10000,
                      help='Maximum sequence length (default: 10000)')
    parser.add_argument('--use_sequential', action='store_true',
                      help='Use sequential masking instead of random masking')

    sequential_group = parser.add_argument_group('Sequential masking parameters')
    sequential_group.add_argument('--sequence_length_csv', type=str,
                      help='Path to CSV file containing flight_id to sequence_length mapping')
    sequential_group.add_argument('--mask_length', type=int,
                      help='Length of sequential mask')
    sequential_group.add_argument('--start_point', type=float,
                      help='Starting point for sequential mask as fraction of sequence length')

    random_group = parser.add_argument_group('Random masking parameters')
    random_group.add_argument('--masking_ratio', type=float,
                      help='Proportion of input to mask for random masking')
    random_group.add_argument('--mean_mask_length', type=int,
                      help='Average length of masking subsequences for random masking')

    args = parser.parse_args()

    # Validate masking parameters
    if args.use_sequential:
        if any(param is None for param in [args.sequence_length_csv, args.mask_length, args.start_point]):
            parser.error("When using sequential masking (--use_sequential), the following arguments are required: "
                        "--sequence_length_csv, --mask_length, --start_point")
    else:
        if any(param is None for param in [args.masking_ratio, args.mean_mask_length]):
            parser.error("When using random masking (default), the following arguments are required: "
                        "--masking_ratio, --mean_mask_length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load normalization parameters
    normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()

    # Load test data
    print("Loading test data...")
    test_data, flight_ids, flight_filenames = load_flight_data_with_filenames(args.data_dir)
    feat_dim = test_data.shape[2]
    print(f"Loaded {len(flight_ids)} flights with {feat_dim} features")

    # Get feature names
    feature_names = get_feature_names(args.data_dir)
    if feature_names is None or len(feature_names) != feat_dim:
        print("Warning: Could not extract feature names, using indices instead")
        feature_names = [f"feature_{i}" for i in range(feat_dim)]

    # Find requested feature index
    try:
        feature_idx = feature_names.index(args.feature)
        print(f"Found {args.feature} at feature index: {feature_idx}")
    except ValueError:
        print(f"Error: '{args.feature}' not found in feature names!")
        print(f"Available features: {feature_names}")
        exit(1)

    # Load model
    print(f"Loading BERT model from {args.model_path}...")
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

    # Load sequence lengths if using sequential masking
    sequence_length_map = None
    if args.use_sequential:
        sequence_length_map = load_sequence_lengths(args.sequence_length_csv)

    # Visualize three aircraft side by side
    print(f"\nGenerating visualization for three aircraft with feature: {args.feature}...")
    print(f"Aircraft types: {', '.join(args.aircraft_names)}")

    flights_data = []
    for aircraft_name in args.aircraft_names:
        print(f"\nProcessing {aircraft_name}...")
        flight_id, flight_filename, original, reconstructed, mask = visualize_single_flight(
            model,
            test_data,
            flight_ids,
            flight_filenames,
            feature_idx,
            normalization_params,
            masking_ratio=args.masking_ratio if not args.use_sequential else 0.5,
            mean_mask_length=args.mean_mask_length if not args.use_sequential else 60,
            use_sequential=args.use_sequential,
            mask_length=args.mask_length if args.use_sequential else 10,
            start_point=args.start_point if args.use_sequential else 0.5,
            sequence_length_map=sequence_length_map,
            aircraft_name=aircraft_name,
            device=device
        )
        flights_data.append((flight_id, flight_filename, original, reconstructed, aircraft_name))

    # Plot all three flights side by side
    print("\nCreating combined visualization...")
    plot_three_flights_side_by_side(
        flights_data,
        feature_idx,
        feature_name=args.feature,
        masking_ratio=args.masking_ratio if not args.use_sequential else None,
        mean_mask_length=args.mean_mask_length if not args.use_sequential else None
    )

    print("\nDone!")
