#!/usr/bin/env python3
from datasets import load_dataset

# First check available configs
try:
    ds = load_dataset('username/NGAFID-LOCI-GATS-Data')
    print("Dataset loaded with default config")
except Exception as e:
    print(f"Error with default config: {e}")

    # Try without config
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names('username/NGAFID-LOCI-GATS-Data')
        print(f"Available configs: {configs}")

        ds = load_dataset('username/NGAFID-LOCI-GATS-Data', configs[0])
        print(f"Loaded with config: {configs[0]}")
    except Exception as e2:
        print(f"Could not load dataset: {e2}")
        exit(1)

print('Dataset info:')
print(ds)

print('\nDataset structure:')
for split_name in ds.keys():
    print(f'{split_name}: {len(ds[split_name])} examples')

if 'train' in ds:
    print('\nFirst example keys:')
    print(list(ds['train'][0].keys()))

    print('\nFirst example sample:')
    first_example = ds['train'][0]
    for key, value in first_example.items():
        if hasattr(value, 'shape'):
            print(f'{key}: shape {value.shape}, dtype {value.dtype}')
        else:
            print(f'{key}: {type(value)} - {str(value)[:100]}...')