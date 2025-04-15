import argparse
from pathlib import Path
import pandas as pd

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

    # Get all file paths in the input directory
    flight_paths = [f for f in input_path.glob('*.csv')]
    columns_names_dict = pd.read_json('standard_column_names.json')['columns'].to_dict()

    if cols_filename:
        columns_set = set(pd.read_csv(cols_filename))

    for flight_path in flight_paths:
        flight_data = pd.read_csv(flight_path, na_values=[' NaN', 'NaN', 'NaN '])
        
        # Clean up column names
        rename_dict = {col: columns_names_dict[col] for col in flight_data.columns if col in columns_names_dict}
        flight_data = flight_data.rename(columns=rename_dict)

        # drop flights missing columns
        flight_cols = set(flight_data.columns)
        mutual = flight_cols & columns_set
        if len(mutual) != len(columns_set):
            continue
        
        if drop_length:
            if len(flight_data) < drop_length:
                #don't save flight if it's too short and drop_length is set
                continue
        
        if pad_length:
            if len(flight_data) <= pad_length:
                flight_data = flight_data.reindex(range(pad_length)).ffil()
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
        default='default_columns.csv',
        help='Indicate the columns a flight must contain. Flights with missing columns will be dropped.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args) 