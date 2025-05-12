import os
from pathlib import Path
from collections import Counter
import argparse
from tqdm import tqdm
from tabulate import tabulate

def extract_aircraft_type(filename):
    """
    Extract aircraft type from filename.
    Modify this function based on your specific filename format.
    Example filename format: "C172_flight_123.csv" or "PA28_flight_456.csv"
    """
    # Split the filename by underscore and take the first part as aircraft type
    aircraft_type = filename.split('_')[0]
    return aircraft_type

def count_aircraft_types(directory):
    """
    Count the number of flights for each aircraft type in the given directory.
    """
    directory = Path(directory)
    aircraft_counter = Counter()
    total_flights = 0
    
    # Get all CSV files in the directory and its subdirectories
    flight_files = list(directory.glob('**/*.csv'))
    
    print(f"Found {len(flight_files)} flight files")
    
    for flight_file in tqdm(flight_files, desc="Processing files"):
        try:
            aircraft_type = extract_aircraft_type(flight_file.stem)
            aircraft_counter[aircraft_type] += 1
            total_flights += 1
        except Exception as e:
            print(f"Error processing file {flight_file}: {e}")
    
    return aircraft_counter, total_flights

def main():
    parser = argparse.ArgumentParser(description='Count flights by aircraft type in a directory')
    parser.add_argument('directory', type=str, help='Directory containing flight data files')
    args = parser.parse_args()
    
    print(f"Analyzing flights in directory: {args.directory}")
    aircraft_counts, total_flights = count_aircraft_types(args.directory)
    
    # Prepare table data
    table_data = []
    headers = ["Aircraft Type", "Number of Flights", "Percentage", "Bar Chart"]
    
    for aircraft_type, count in sorted(aircraft_counts.items()):
        percentage = (count / total_flights) * 100
        # Create a simple bar chart using blocks
        bar_length = int(percentage / 2)  # Scale down by 2 to keep bars reasonable
        bar = "█" * bar_length
        
        table_data.append([
            aircraft_type,
            count,
            f"{percentage:.1f}%",
            bar
        ])
    
    # Add total row
    table_data.append([
        "TOTAL",
        total_flights,
        "100.0%",
        "█" * 50
    ])
    
    # Print the table
    print("\nAircraft Type Distribution:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main() 