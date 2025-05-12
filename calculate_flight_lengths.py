import os
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def get_flight_id(filename):
    """
    Extract flight ID from filename between underscore and .csv
    Example: something_12345.csv -> 12345
    """
    try:
        return filename.split('_')[-1].split('.csv')[0]
    except Exception as e:
        print(f"Error extracting flight ID from {filename}: {str(e)}")
        return None

def calculate_flight_length(csv_file):
    """
    Calculate the length of a flight from its CSV file.
    Length is determined by the number of rows in the CSV.
    """
    try:
        df = pd.read_csv(csv_file)
        return len(df)
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        return None

def process_directory(input_dir, output_file):
    """
    Process all CSV files in the input directory and create a mapping
    of flight IDs to their lengths.
    """
    input_path = Path(input_dir)
    flight_lengths = {}

    # Check if directory exists
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")

    # Get list of CSV files first
    csv_files = list(input_path.glob("*.csv"))
    
    # Process each CSV file with progress bar
    for csv_file in tqdm(csv_files, desc="Processing flights", unit="file"):
        flight_id = get_flight_id(csv_file.name)  # Extract ID from filename
        if flight_id is not None:
            length = calculate_flight_length(csv_file)
            if length is not None:
                flight_lengths[flight_id] = length

    # Create DataFrame and save to CSV
    df = pd.DataFrame.from_dict(flight_lengths, orient='index', columns=['length'])
    df.index.name = 'flight_id'
    df.to_csv(output_file)
    print(f"Flight lengths have been saved to {output_file}")
    print(f"Processed {len(flight_lengths)} flights")

def main():
    parser = argparse.ArgumentParser(description='Calculate lengths of flights from CSV files')
    parser.add_argument('input_dir', help='Directory containing flight CSV files')
    parser.add_argument('--output', '-o', default='flight_lengths.csv',
                      help='Output CSV file (default: flight_lengths.csv)')
    
    args = parser.parse_args()
    
    try:
        process_directory(args.input_dir, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 