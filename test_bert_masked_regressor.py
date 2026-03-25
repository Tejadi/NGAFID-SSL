import torch
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from ngafid_datasets.transformation_dataset import mask_transform, sequential_mask_transform
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import load_flight_data, plot_aircraft_type_comparison, plot_reconstructions, get_aircraft_counts, load_sequence_lengths, plot_sequential_reconstructions
from models.bert_masked_regressor import BertMaskedRegressor

def load_anomaly_labels(labels_csv, events_csv=None):
    """Load binary anomaly labels per flight.

    Supports two formats:
      1. A labels CSV with columns: flight_id, label  (1 = anomalous, 0 = normal)
      2. An events CSV (NGAFID format) with columns including flight_id.
         Any flight that appears in the events file is labelled anomalous.
         Requires a flight_ids CSV to know the full set of flights.
    """
    if labels_csv is not None:
        df = pd.read_csv(labels_csv)
        if 'label' not in df.columns or 'flight_id' not in df.columns:
            raise ValueError("labels_csv must have columns: flight_id, label")
        return dict(zip(df['flight_id'], df['label'].astype(int)))

    if events_csv is not None:
        events = pd.read_csv(events_csv)
        anomalous_ids = set(events['flight_id'].unique())
        return anomalous_ids  # caller will use set membership

    raise ValueError("Provide either --labels_csv or --events_csv for anomaly evaluation")


def compute_topk_metrics(scores, labels, ks=(0.01, 0.02, 0.05, 0.10)):
    """Compute precision@k and recall@k for top-k anomaly retrieval.

    Args:
        scores: np.array of anomaly scores (higher = more anomalous), shape (n,)
        labels: np.array of binary labels (1 = anomalous), shape (n,)
        ks: tuple of floats in (0,1] interpreted as percentages of n,
            or ints > 1 interpreted as absolute counts.

    Returns:
        list of dicts with keys: k_pct, k_abs, precision, recall, num_retrieved, num_true_in_topk, total_anomalies
    """
    n = len(scores)
    total_anomalies = int(labels.sum())
    # Sort descending by score
    ranked_indices = np.argsort(scores)[::-1]
    ranked_labels = labels[ranked_indices]

    results = []
    for k in ks:
        if isinstance(k, float) and k <= 1.0:
            k_abs = max(1, int(np.ceil(k * n)))
            k_pct = k
        else:
            k_abs = int(k)
            k_pct = k_abs / n

        topk_labels = ranked_labels[:k_abs]
        num_true = int(topk_labels.sum())
        precision = num_true / k_abs if k_abs > 0 else 0.0
        recall = num_true / total_anomalies if total_anomalies > 0 else 0.0

        results.append({
            'k_pct': f"{k_pct:.1%}",
            'k_abs': k_abs,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'num_true_in_topk': num_true,
            'num_retrieved': k_abs,
            'total_anomalies': total_anomalies,
        })

    return results


def run_anomaly_eval(flight_scores, labels_csv=None, events_csv=None,
                     ks=(0.01, 0.02, 0.05, 0.10), output_path='topk_anomaly_metrics.json'):
    """Full anomaly evaluation: ROC-AUC + top-k retrieval metrics.

    Args:
        flight_scores: DataFrame with columns flight_id, anomaly_score
        labels_csv: path to CSV with flight_id,label columns
        events_csv: path to NGAFID events CSV (alternative label source)
        ks: k values for top-k metrics
        output_path: where to save JSON results
    """
    label_source = load_anomaly_labels(labels_csv, events_csv)

    if isinstance(label_source, set):
        # events-based: membership = anomalous
        flight_scores['label'] = flight_scores['flight_id'].apply(
            lambda fid: 1 if fid in label_source else 0)
    else:
        # dict-based
        flight_scores['label'] = flight_scores['flight_id'].map(label_source)
        missing = flight_scores['label'].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} flights have no label and will be dropped")
            flight_scores = flight_scores.dropna(subset=['label'])
        flight_scores['label'] = flight_scores['label'].astype(int)

    scores = flight_scores['anomaly_score'].values
    labels = flight_scores['label'].values

    total = len(labels)
    n_pos = int(labels.sum())
    n_neg = total - n_pos
    print(f"\nAnomaly evaluation: {total} flights, {n_pos} anomalous, {n_neg} normal")

    # ROC-AUC
    if n_pos == 0 or n_neg == 0:
        print("Warning: Only one class present, ROC-AUC is undefined")
        auc = None
    else:
        auc = roc_auc_score(labels, scores)
        print(f"ROC-AUC: {auc:.4f}")

    # Top-k retrieval
    topk_results = compute_topk_metrics(scores, labels, ks)

    print(f"\n{'k':<10} {'k_abs':<8} {'Prec@k':<10} {'Rec@k':<10} {'#True':<8} {'#Ret':<8}")
    print("-" * 54)
    for r in topk_results:
        print(f"{r['k_pct']:<10} {r['k_abs']:<8} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['num_true_in_topk']:<8} {r['num_retrieved']:<8}")

    # Save results
    output = {
        'roc_auc': auc,
        'total_flights': total,
        'total_anomalies': n_pos,
        'total_normal': n_neg,
        'topk_metrics': topk_results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


def load_bert_model(model_path, feat_dim, hidden_size, encoder_layers, decoder_layers, num_heads, max_seq_len, device):
    """Load trained BERT masked regressor model."""
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=False,  # Disable during inference
        use_mixed_precision=False,  # Disable during inference
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_data, flight_ids, normalization_params, batch_size=32, masking_ratio=0.5, mean_mask_length=60,
                  device="cuda" if torch.cuda.is_available() else "cpu"):

    model.eval()

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    test_dataset = TensorDataset(torch.FloatTensor(test_data_normalized), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    all_masks = []
    per_flight_mse = []
    per_flight_ids = []

    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)

            original_data = data.cpu().numpy()
            masked_batch = []
            batch_masks = []
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
                batch_masks.append(mask.numpy())

            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)

            reconstructed = model(masked_data)

            # Compute metrics on normalized values (as per experiment description)
            original_norm = data.cpu().numpy()
            recon_norm = reconstructed.cpu().numpy()

            mae = np.mean(np.abs(original_norm - recon_norm))
            mse = np.mean((original_norm - recon_norm) ** 2)

            # Per-flight MSE: mean over (seq_len, feat_dim) for each sample
            flight_mses = np.mean((original_norm - recon_norm) ** 2, axis=(1, 2))
            per_flight_mse.extend(flight_mses.tolist())
            per_flight_ids.extend(batch_ids.numpy().tolist())

            total_mae += mae
            total_mse += mse
            num_batches += 1

            # Denormalize for visualization only
            original_denorm = original_norm * normalization_params['std'] + normalization_params['mean']
            recon_denorm = recon_norm * normalization_params['std'] + normalization_params['mean']
            all_orig.append(original_denorm)
            all_recon.append(recon_denorm)
            all_masks.append(np.stack(batch_masks, axis=0))

    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    rmse = np.sqrt(avg_mse)

    metrics = {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': rmse
    }

    flight_scores = pd.DataFrame({
        'flight_id': per_flight_ids,
        'anomaly_score': per_flight_mse
    })

    return metrics, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_masks), flight_scores

def evaluate_sequential_model(model, test_data, flight_ids, sequence_length_map, normalization_params, batch_size=32,
                            mask_length=10, start_point=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    flight_ids_list = list(flight_ids)
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data_normalized),
        torch.LongTensor([sequence_length_map[id] for id in flight_ids]),
        torch.LongTensor(flight_ids_list)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    all_masks = []
    per_flight_mse = []
    per_flight_ids = []

    with torch.no_grad():
        for data, seq_lengths, batch_ids in tqdm(test_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)

            original_data = data.cpu().numpy()
            masked_batch = []
            batch_masks = []
            for sequence, seq_len in zip(original_data, seq_lengths):
                _, masked_sequence, mask = sequential_mask_transform(
                    sequence,
                    starting_point=start_point,
                    n=mask_length,
                    sequence_length=seq_len.item()
                )
                masked_sequence = masked_sequence.numpy()
                masked_batch.append(masked_sequence)
                batch_masks.append(mask.numpy())

            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)

            reconstructed = model(masked_data)

            # Compute metrics on normalized values (as per experiment description)
            original_norm = data.cpu().numpy()
            recon_norm = reconstructed.cpu().numpy()

            mae = np.mean(np.abs(original_norm - recon_norm))
            mse = np.mean((original_norm - recon_norm) ** 2)

            # Per-flight MSE: mean over (seq_len, feat_dim) for each sample
            flight_mses = np.mean((original_norm - recon_norm) ** 2, axis=(1, 2))
            per_flight_mse.extend(flight_mses.tolist())
            per_flight_ids.extend(batch_ids.numpy().tolist())

            total_mae += mae
            total_mse += mse
            num_batches += 1

            # Denormalize for visualization only
            original_denorm = original_norm * normalization_params['std'] + normalization_params['mean']
            recon_denorm = recon_norm * normalization_params['std'] + normalization_params['mean']
            all_orig.append(original_denorm)
            all_recon.append(recon_denorm)
            all_masks.append(np.stack(batch_masks, axis=0))

    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    rmse = np.sqrt(avg_mse)

    metrics = {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': rmse
    }

    flight_scores = pd.DataFrame({
        'flight_id': per_flight_ids,
        'anomaly_score': per_flight_mse
    })

    return metrics, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_masks), flight_scores

def evaluate_model_per_feature(model, test_data, flight_ids, normalization_params, batch_size=32, masking_ratio=0.5, mean_mask_length=60,
                                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate model and compute per-feature metrics.
    Returns MSE and MAE for each feature individually.
    """
    model.eval()

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    test_dataset = TensorDataset(torch.FloatTensor(test_data_normalized), torch.LongTensor(flight_ids))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    feat_dim = test_data.shape[2]

    # Accumulate per-feature errors
    per_feature_mse = np.zeros(feat_dim)
    per_feature_mae = np.zeros(feat_dim)
    per_feature_samples = np.zeros(feat_dim)

    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating per-feature metrics", unit="batch"):
            data = data.to(device)

            original_data = data.cpu().numpy()
            masked_batch = []
            batch_masks = []
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
                batch_masks.append(mask.numpy())

            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)

            reconstructed = model(masked_data)

            # Compute metrics on normalized values
            original_norm = data.cpu().numpy()
            recon_norm = reconstructed.cpu().numpy()

            # Compute per-feature metrics
            # Shape: (batch_size, seq_len, feat_dim)
            for feat_idx in range(feat_dim):
                feat_errors = original_norm[:, :, feat_idx] - recon_norm[:, :, feat_idx]
                per_feature_mse[feat_idx] += np.sum(feat_errors ** 2)
                per_feature_mae[feat_idx] += np.sum(np.abs(feat_errors))
                per_feature_samples[feat_idx] += original_norm[:, :, feat_idx].size

    # Average over all samples
    per_feature_mse = per_feature_mse / per_feature_samples
    per_feature_mae = per_feature_mae / per_feature_samples
    per_feature_rmse = np.sqrt(per_feature_mse)

    return per_feature_mse, per_feature_mae, per_feature_rmse

def evaluate_sequential_model_per_feature(model, test_data, flight_ids, sequence_length_map, normalization_params, batch_size=32,
                                          mask_length=10, start_point=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate sequential model and compute per-feature metrics.
    Returns MSE and MAE for each feature individually.
    """
    model.eval()

    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']

    test_dataset = TensorDataset(
        torch.FloatTensor(test_data_normalized),
        torch.LongTensor([sequence_length_map[id] for id in flight_ids])
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    feat_dim = test_data.shape[2]

    # Accumulate per-feature errors
    per_feature_mse = np.zeros(feat_dim)
    per_feature_mae = np.zeros(feat_dim)
    per_feature_samples = np.zeros(feat_dim)

    with torch.no_grad():
        for data, seq_lengths in tqdm(test_loader, desc="Evaluating per-feature metrics", unit="batch"):
            data = data.to(device)

            original_data = data.cpu().numpy()
            masked_batch = []
            batch_masks = []
            for sequence, seq_len in zip(original_data, seq_lengths):
                _, masked_sequence, mask = sequential_mask_transform(
                    sequence,
                    starting_point=start_point,
                    n=mask_length,
                    sequence_length=seq_len.item()
                )
                masked_sequence = masked_sequence.numpy()
                masked_batch.append(masked_sequence)
                batch_masks.append(mask.numpy())

            masked_data = np.stack(masked_batch, axis=0)
            masked_data = torch.FloatTensor(masked_data).to(device)

            reconstructed = model(masked_data)

            # Compute metrics on normalized values
            original_norm = data.cpu().numpy()
            recon_norm = reconstructed.cpu().numpy()

            # Compute per-feature metrics
            # Shape: (batch_size, seq_len, feat_dim)
            for feat_idx in range(feat_dim):
                feat_errors = original_norm[:, :, feat_idx] - recon_norm[:, :, feat_idx]
                per_feature_mse[feat_idx] += np.sum(feat_errors ** 2)
                per_feature_mae[feat_idx] += np.sum(np.abs(feat_errors))
                per_feature_samples[feat_idx] += original_norm[:, :, feat_idx].size

    # Average over all samples
    per_feature_mse = per_feature_mse / per_feature_samples
    per_feature_mae = per_feature_mae / per_feature_samples
    per_feature_rmse = np.sqrt(per_feature_mse)

    return per_feature_mse, per_feature_mae, per_feature_rmse

def get_feature_names(data_dir):
    """
    Extract feature names from the first CSV file in the data directory.
    """
    # Try to find a sample CSV file
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        csv_files = [os.path.join(root, f) for f in files if f.endswith('.csv') and
                     not any(name in f.lower() for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]
        if csv_files:
            break

    if not csv_files:
        return None

    # Read the first CSV to get column names
    try:
        df = pd.read_csv(csv_files[0], nrows=0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    except Exception as e:
        print(f"Warning: Could not extract feature names: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained BERT masked regressor on flight data')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test flight CSV files')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights')
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
    parser.add_argument('--use_sequential', action='store_true',
                      help='Use sequential masking instead of random masking')
    parser.add_argument('--per_feature_analysis', action='store_true',
                      help='Compute and save per-feature MSE and MAE metrics to CSV')
    parser.add_argument('--output_csv', type=str, default='per_feature_metrics.csv',
                      help='Output CSV file for per-feature metrics (default: per_feature_metrics.csv)')

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

    anomaly_group = parser.add_argument_group('Anomaly detection evaluation')
    anomaly_group.add_argument('--anomaly_eval', action='store_true',
                      help='Run top-k anomaly retrieval evaluation')
    anomaly_group.add_argument('--labels_csv', type=str,
                      help='CSV with columns: flight_id, label (1=anomalous, 0=normal)')
    anomaly_group.add_argument('--events_csv', type=str,
                      help='NGAFID events CSV (flights with events are anomalous)')
    anomaly_group.add_argument('--topk_output', type=str, default='topk_anomaly_metrics.json',
                      help='Output JSON file for top-k metrics (default: topk_anomaly_metrics.json)')
    anomaly_group.add_argument('--topk_values', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.10],
                      help='Top-k percentages to evaluate (default: 0.01 0.02 0.05 0.10)')

    args = parser.parse_args()


    if args.use_sequential:
        if any(param is None for param in [args.sequence_length_csv, args.mask_length, args.start_point]):
            parser.error("When using sequential masking (--use_sequential), the following arguments are required: "
                        "--sequence_length_csv, --mask_length, --start_point")
    else:
        if any(param is None for param in [args.masking_ratio, args.mean_mask_length]):
            parser.error("When using random masking (default), the following arguments are required: "
                        "--masking_ratio, --mean_mask_length")

    if args.anomaly_eval and args.labels_csv is None and args.events_csv is None:
        parser.error("--anomaly_eval requires either --labels_csv or --events_csv")

    print("Analyzing aircraft types in the data directory...")
    aircraft_counts = get_aircraft_counts(args.data_dir)
    print(f"Found aircraft counts: {aircraft_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()

    test_data, flight_ids = load_flight_data(args.data_dir)
    feat_dim = test_data.shape[2]

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

    print("Evaluating model...")
    if args.use_sequential:
        sequence_length_map = load_sequence_lengths(args.sequence_length_csv)
        metrics, orig_data, recon_data, masks, flight_scores = evaluate_sequential_model(
            model,
            test_data,
            flight_ids,
            sequence_length_map,
            normalization_params=normalization_params,
            batch_size=args.batch_size,
            mask_length=args.mask_length,
            start_point=args.start_point,
            device=device
        )
    else:
        metrics, orig_data, recon_data, masks, flight_scores = evaluate_model(
            model,
            test_data,
            flight_ids,
            normalization_params=normalization_params,
            batch_size=args.batch_size,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            device=device
        )

    print("\nTest Metrics (Overall):")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")

    # Anomaly detection evaluation
    if args.anomaly_eval:
        print("\n" + "="*60)
        print("Running anomaly detection evaluation (top-k retrieval)")
        print("="*60)
        run_anomaly_eval(
            flight_scores,
            labels_csv=args.labels_csv,
            events_csv=args.events_csv,
            ks=tuple(args.topk_values),
            output_path=args.topk_output,
        )

    # Per-feature analysis
    if args.per_feature_analysis:
        print("\n" + "="*60)
        print("Computing per-feature metrics...")
        print("="*60)

        if args.use_sequential:
            sequence_length_map = load_sequence_lengths(args.sequence_length_csv)
            per_feat_mse, per_feat_mae, per_feat_rmse = evaluate_sequential_model_per_feature(
                model,
                test_data,
                flight_ids,
                sequence_length_map,
                normalization_params=normalization_params,
                batch_size=args.batch_size,
                mask_length=args.mask_length,
                start_point=args.start_point,
                device=device
            )
        else:
            per_feat_mse, per_feat_mae, per_feat_rmse = evaluate_model_per_feature(
                model,
                test_data,
                flight_ids,
                normalization_params=normalization_params,
                batch_size=args.batch_size,
                masking_ratio=args.masking_ratio,
                mean_mask_length=args.mean_mask_length,
                device=device
            )

        # Get feature names
        feature_names = get_feature_names(args.data_dir)
        if feature_names is None or len(feature_names) != feat_dim:
            print("Warning: Could not extract feature names, using indices instead")
            feature_names = [f"feature_{i}" for i in range(feat_dim)]

        # Create results dataframe
        results_df = pd.DataFrame({
            'feature_index': list(range(feat_dim)),
            'feature_name': feature_names,
            'mse': per_feat_mse,
            'mae': per_feat_mae,
            'rmse': per_feat_rmse
        })

        # Sort by MSE to see best/worst features
        results_df_sorted = results_df.sort_values('mse')

        # Save to CSV
        results_df_sorted.to_csv(args.output_csv, index=False)
        print(f"\n✓ Per-feature metrics saved to: {args.output_csv}")

        # Print summary statistics
        print("\nPer-Feature Metrics Summary:")
        print(f"{'Feature':<30} {'Index':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
        print("-" * 80)

        # Print top 10 best features (lowest MSE)
        print("\nTop 10 Best Reconstructed Features (Lowest MSE):")
        for idx, row in results_df_sorted.head(10).iterrows():
            print(f"{row['feature_name']:<30} {int(row['feature_index']):<8} {row['mse']:<12.6f} {row['mae']:<12.6f} {row['rmse']:<12.6f}")

        print("\nTop 10 Worst Reconstructed Features (Highest MSE):")
        for idx, row in results_df_sorted.tail(10).iterrows():
            print(f"{row['feature_name']:<30} {int(row['feature_index']):<8} {row['mse']:<12.6f} {row['mae']:<12.6f} {row['rmse']:<12.6f}")

        print(f"\nOverall Statistics:")
        print(f"  Mean MSE across features: {per_feat_mse.mean():.6f}")
        print(f"  Mean MAE across features: {per_feat_mae.mean():.6f}")
        print(f"  Std MSE across features: {per_feat_mse.std():.6f}")
        print(f"  Std MAE across features: {per_feat_mae.std():.6f}")

    print("\nGenerating reconstruction plots...")
    if args.use_sequential:
        plot_sequential_reconstructions(
            orig_data,
            recon_data,
            flight_ids,
            feature_indices=[34],
            start_point=args.start_point,
            mask_length=args.mask_length,
            sequence_length_csv=args.sequence_length_csv,
            num_samples=5
        )
        print("Sequential reconstruction plots saved as 'sequential_reconstruction_flight_X_feature_Y.png'")
    else:
        plot_aircraft_type_comparison(orig_data, recon_data, aircraft_counts)
        print("Aircraft type comparison plots saved as 'aircraft_comparison_feature_X.png' for each feature")
