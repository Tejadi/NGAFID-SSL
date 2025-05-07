import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def main(args):
    """
    Main function to process the data.
    
    Args:
        args: Parsed command line arguments
    """
    input_path = Path(args.input)
    output_path = Path(args.output)
    pad_length = args.pad
    drop_length = args.drop
    na_strategy = args.na
    cols_filename = args.cols
    delete_original = args.delete_original

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all file paths in the input directory
    flight_paths = [f for f in input_path.glob('*.csv')]

    if cols_filename:
        with open(cols_filename, 'r') as f:
            columns_set = set([line.strip().replace(' ', '').lower() for line in f if line.strip()])
    
    for flight_path in tqdm(flight_paths, desc="Processing flights"):
        try:
            flight_data = pd.read_csv(flight_path, na_values=[' NaN', 'NaN', 'NaN '])

            if cols_filename:
                # drop flights missing columns
                flight_cols = set([column.strip().replace(' ', '').lower() for column in flight_data.columns])
                mutual = flight_cols & columns_set
                if len(mutual) != len(columns_set):
                    if delete_original:
                        os.remove(flight_path)
                    continue
        
            flight_data.columns = [col.strip().replace(' ', '').lower() for col in flight_data.columns]
            flight_data = flight_data[list(columns_set)]

            if drop_length:
                if len(flight_data) < drop_length:
                    if delete_original:
                        os.remove(flight_path)
                    continue
            
            if pad_length:
                if len(flight_data) <= pad_length:
                    flight_data = flight_data.reindex(range(pad_length)).ffill()
                else:
                    if delete_original:
                        os.remove(flight_path)
                    continue

            if na_strategy == 'zero':
                flight_data = flight_data.fillna(0)
            else:
                flight_data = flight_data.ffill().bfill() #bfill to fill first row of NA values
            
            for col in flight_data.select_dtypes(include='object').columns:
                flight_data[col] = pd.to_numeric(flight_data[col], errors='coerce')

            
            # Save processed file
            output_file = output_path / flight_path.name
            flight_data.to_csv(output_file, index=False)
            
            # Delete original file if flag is set and output file exists
            if delete_original and output_file.exists():
                os.remove(flight_path)
                
        except Exception as e:
            print(f"Error processing {flight_path}: {str(e)}")
            if delete_original:
                os.remove(flight_path)
                
    if args.split:
        processed_files = list(output_path.glob('*.csv'))
            
        # Shuffle file paths
        np.random.shuffle(processed_files)
        
        # Calculate split indices
        train_size = int(len(processed_files) * args.train_ratio)
        val_size = int(len(processed_files) * args.val_ratio)

        train_files = processed_files[:train_size]
        val_files = processed_files[train_size:train_size + val_size]
        test_files = processed_files[train_size+val_size:]

        # Create directories for splits
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        test_dir = output_path / 'test'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # Move files to split directories
        for file in train_files:
            file.rename(train_dir / file.name)
        for file in val_files:
            file.rename(val_dir / file.name)
        for file in test_files:
            file.rename(test_dir / file.name)

        print(f"Data split into train ({len(train_files)}), val ({len(val_files)}), and test ({len(test_files)}) sets.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Data preprocessing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input directory containing data files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '-pad',
        type=int,
        action='store',
        help='Pad data to fixed length. Flights with more than this length will be dropped.'
    )
    parser.add_argument(
        '-drop',
        type=int,
        action='store',
        help='Drop flights with less than this many rows.'
    )
    parser.add_argument(
        '-na',
        type=str,
        choices=['zero', 'ffil'],
        default='zero',
        help='Indicate how to handle NA values in the flight.'
    )
    parser.add_argument(
        '-cols',
        type=str,
        default='preprocessing/default_columns.txt',
        help='Indicate the columns a flight must contain. Flights with missing columns will be dropped. Provide a txt filename with newline separated column names'
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        help='Delete original files after preprocessing (including files that fail preprocessing criteria)'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Enable data splitting into train, val, and test sets.'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of data to be used for training.'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of data to be used for validation.'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of data to be used for testing.'
    )

    args = parser.parse_args()
    # Validate that the sum of train, val, and test ratios equals 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"The sum of train, val, and test ratios must be 1.0, but is {total_ratio}.")
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args) 