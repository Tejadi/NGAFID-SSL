import pandas as pd
import os
from pathlib import Path

def analyze_flight_data(directory_path):
    """
    Read all CSV files in the specified directory and analyze their columns.
    
    Args:
        directory_path (str): Path to the directory containing CSV files
    """
    # Initialize a set to store all unique columns
    all_columns = set()
    
    # Get all CSV files in the directory
    csv_files = [f for f in Path(directory_path).glob('*.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the columns
            columns = set(df.columns)
            
            # Add to the set of all columns
            all_columns.update(columns)
            
            # Print file information
            # print(f"\nFile: {file_path.name}")
            # print(f"Number of rows: {len(df)}")
            # print(f"Columns ({len(columns)}):")
            # for col in sorted(columns):
            #     print(f"  - {col}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Print summary of all unique columns
    print("\nSummary of all unique columns across all files:")
    print(f"Total unique columns: {len(all_columns)}")
    for col in sorted(all_columns):
        print(f"  - {col}")

if __name__ == "__main__":
    # Specify the directory containing the flight data
    directory_path = "../../loci_dataset"  # Update this path as needed
    
    try:
        analyze_flight_data(directory_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}") 
    