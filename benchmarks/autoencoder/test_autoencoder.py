import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datasets.transformation_dataset import mask_transform, sequential_mask_transform
import argparse
from tqdm import tqdm
from utils import load_model, load_flight_data, plot_aircraft_type_comparison, plot_reconstructions, get_aircraft_counts, load_sequence_lengths, plot_sequential_reconstructions

def evaluate_model(model, test_data, flight_ids, normalization_params, batch_size=32, masking_ratio=0.6, mean_mask_length=3,
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
            
            original_denorm = data.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            recon_denorm = reconstructed.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            
            mae = np.mean(np.abs(original_denorm - recon_denorm))
            mse = np.mean((original_denorm - recon_denorm) ** 2)
            
            total_mae += mae
            total_mse += mse
            num_batches += 1
            
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
    
    return metrics, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_masks)

def evaluate_sequential_model(model, test_data, flight_ids, sequence_length_map, normalization_params, batch_size=32, 
                            mask_length=10, start_point=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    
    test_data_normalized = (test_data - normalization_params['mean']) / normalization_params['std']
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data_normalized),
        torch.LongTensor([sequence_length_map[id] for id in flight_ids])
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    total_mae = 0
    total_mse = 0
    num_batches = 0
    all_orig = []
    all_recon = []
    all_masks = []
    
    with torch.no_grad():
        for data, seq_lengths in tqdm(test_loader, desc="Evaluating", unit="batch"):
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
            
            original_denorm = data.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            recon_denorm = reconstructed.cpu().numpy() * normalization_params['std'] + normalization_params['mean']
            
            mae = np.mean(np.abs(original_denorm - recon_denorm))
            mse = np.mean((original_denorm - recon_denorm) ** 2)
            
            total_mae += mae
            total_mse += mse
            num_batches += 1
            
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
    
    return metrics, np.concatenate(all_orig), np.concatenate(all_recon), np.concatenate(all_masks)

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
    

    if args.use_sequential:
        if any(param is None for param in [args.sequence_length_csv, args.mask_length, args.start_point]):
            parser.error("When using sequential masking (--use_sequential), the following arguments are required: "
                        "--sequence_length_csv, --mask_length, --start_point")
    else:
        if any(param is None for param in [args.masking_ratio, args.mean_mask_length]):
            parser.error("When using random masking (default), the following arguments are required: "
                        "--masking_ratio, --mean_mask_length")
    
    print("Analyzing aircraft types in the data directory...")
    aircraft_counts = get_aircraft_counts(args.data_dir)
    print(f"Found aircraft counts: {aircraft_counts}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()
    
    test_data, flight_ids = load_flight_data(args.data_dir)
    input_dim = test_data.shape[2]
    
    model = load_model(args.model_path, input_dim, args.hidden_dim, device)
    
    print("Evaluating model...")
    if args.use_sequential:
        sequence_length_map = load_sequence_lengths(args.sequence_length_csv)
        metrics, orig_data, recon_data, masks = evaluate_sequential_model(
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
        metrics, orig_data, recon_data, masks = evaluate_model(
            model,
            test_data,
            flight_ids,
            normalization_params=normalization_params,
            batch_size=args.batch_size,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            device=device
        )
    
    print("\nTest Metrics:")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    
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