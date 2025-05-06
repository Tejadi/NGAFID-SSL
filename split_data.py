import os
import shutil
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Define split ratios
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1

def create_directories():
    """Create train, test, and validation directories if they don't exist."""
    for split in ['train', 'test', 'val']:
        os.makedirs(f'preprocessed_data_{split}', exist_ok=True)

def get_aircraft_files():
    """Group files by aircraft type."""
    aircraft_files = defaultdict(list)
    
    for filename in os.listdir('preprocessed_data'):
        if filename.endswith('.csv'):
            aircraft_type = filename.split('_flight_')[0]
            aircraft_files[aircraft_type].append(filename)
    
    return aircraft_files

def split_files(files, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """Split files into train, test, and validation sets."""
    random.shuffle(files)
    n_files = len(files)
    
    train_size = int(n_files * train_ratio)
    test_size = int(n_files * test_ratio)
    
    train_files = files[:train_size]
    test_files = files[train_size:train_size + test_size]
    val_files = files[train_size + test_size:]
    
    return train_files, test_files, val_files

def copy_files(file_list, destination_dir):
    """Copy files to their respective directories."""
    for filename in file_list:
        src = os.path.join('preprocessed_data', filename)
        dst = os.path.join(destination_dir, filename)
        shutil.copy2(src, dst)

def main():
    # Create output directories
    create_directories()
    
    # Get files grouped by aircraft type
    aircraft_files = get_aircraft_files()
    
    # Split and copy files for each aircraft type
    for aircraft_type, files in aircraft_files.items():
        print(f"\nProcessing {aircraft_type}:")
        train_files, test_files, val_files = split_files(files)
        
        print(f"Train files: {len(train_files)}")
        print(f"Test files: {len(test_files)}")
        print(f"Validation files: {len(val_files)}")
        
        # Copy files to their respective directories
        copy_files(train_files, 'preprocessed_data_train')
        copy_files(test_files, 'preprocessed_data_test')
        copy_files(val_files, 'preprocessed_data_val')

if __name__ == "__main__":
    main() 