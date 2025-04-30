import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

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

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all file paths in the input directory
    flight_paths = [f for f in input_path.glob('*.csv')]

    if cols_filename:
        with open(cols_filename, 'r') as f:
            columns_set = set([line.strip().replace(' ', '').lower() for line in f if line.strip()])

    for flight_path in tqdm(flight_paths, desc="Processing flights"):
        flight_data = pd.read_csv(flight_path, na_values=[' NaN', 'NaN', 'NaN '])

        # drop flights missing columns
        flight_cols = set([column.strip().replace(' ', '').lower() for column in flight_data.columns])
        mutual = flight_cols & columns_set
        if len(mutual) != len(columns_set):
            continue
        
        # Create a mapping from lowercase column names to original column names
        col_mapping = {col.strip().replace(' ', '').lower(): col for col in flight_data.columns}
        # Keep only the columns from columns_set
        flight_data = flight_data[[col_mapping[col] for col in columns_set]]
        
        if drop_length:
            if len(flight_data) < drop_length:
                #don't save flight if it's too short and drop_length is set
                continue
        
        if pad_length:
            if len(flight_data) <= pad_length:
                flight_data = flight_data.reindex(range(pad_length)).ffill()
            else:
                #don't save flight if it's too long and pad_length is set
                continue

        if na_strategy == 'zero':
            flight_data = flight_data.fillna(0)
        else:
            flight_data = flight_data.ffill().bfill() #bfill to fill first row of NA values
        
        flight_data.to_csv(output_path / flight_path.name, index=False)

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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args) 