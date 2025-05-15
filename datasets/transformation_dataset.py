import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def sequential_mask_transform(X, starting_point, n, sequence_length=None):
    if sequence_length is None:
        sequence_length = X.shape[0]
    start_idx = int(sequence_length * starting_point)
    pad_to = min(sequence_length, start_idx + n)
    
    mask = np.ones_like(X, dtype=bool)
    
    mask[start_idx:pad_to, :] = False
    
    X = torch.from_numpy(X)
    mask = torch.from_numpy(mask)
    transformed_X = X * mask
    
    return X, transformed_X, mask

def mask_transform(X, masking_ratio=0.6, mean_mask_length=3, mode='separate', distribution='geometric', exclude_feats=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    mask = noise_mask(X, masking_ratio, mean_mask_length, mode, distribution, exclude_feats, random_seed)  # (seq_length, feat_dim) boolean array
    X = torch.from_numpy(X)
    mask = torch.from_numpy(mask)
    transformed_X = X * mask
    return X, transformed_X, mask

# Credit: Adapted from Repo mvts_transformer by Author George Zerveas from: https://github.com/gzerveas/mvts_transformer
# Original file: src/datasets/dataset.py, licensed under MIT
def noise_mask(X, masking_ratio, lm, mode, distribution, exclude_feats, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
        
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':
        if mode == 'separate':
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio, random_seed)
        else:
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio, random_seed), 1), X.shape[1])
    else:
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    return mask

# Credit: Adapted from Repo mvts_transformer by Author George Zerveas from: https://github.com/gzerveas/mvts_transformer
# Original file: src/datasets/dataset.py, licensed under MIT
def geom_noise_mask_single(L, lm, masking_ratio, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def noise_transform(X, loc=0, range=(0.1, 0.5), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    mean = X.mean(axis=0)
    std_dev = X.std(axis=0)
    
    std_dev_safe = np.where(std_dev == 0, 1, std_dev)
    X_standardized = (X - mean) / std_dev_safe

    deviation = np.random.uniform(range[0], range[1])
    noise = np.random.normal(loc, deviation, X.shape)
    
    noise[:, std_dev == 0] = 0
    X_transformed = X_standardized + noise

    return X_standardized, X_transformed

class TransformationDataset(Dataset):
    def __init__(self, flight_id_topath, transformation_method=noise_transform):
        self.flight_id_topath = flight_id_topath.reset_index()
        self.transformation = transformation_method
    
    def __len__(self):
        return len(self.flight_id_topath)
    
    def __getitem__(self, index):
        path = self.flight_id_topath.loc[index, "file_path"]
        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN '])
        flight_T = flight.T
        flight_T.ffill(inplace= True, axis=0)
        flight_T.bfill(inplace= True, axis=0)
        flight = flight_T.T
        flight = flight.to_numpy()
        flight, flight_transformed = self.transformation(flight)
        flight_transformed = torch.tensor(flight_transformed, dtype=torch.float32)
        flight = torch.tensor(flight, dtype=torch.float32)
        pos_pair = (flight.unsqueeze(dim=0), flight_transformed.unsqueeze(dim=0))
        return pos_pair

class TransformationDatasetReverse(Dataset):
    def __init__(self, flight_id_topath, transformation_method=noise_transform, reverse_masked=False, reverse_original=False):
        self.flight_id_topath = flight_id_topath.reset_index()
        self.transformation = transformation_method
        self.reverse_masked = reverse_masked
        self.reverse_original = reverse_original
    
    def __len__(self):
        return len(self.flight_id_topath)
    
    def __getitem__(self, index):
        path = self.flight_id_topath.loc[index, "file_path"]
        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN '])
        flight_T = flight.T
        flight_T.ffill(inplace= True, axis=0)
        flight_T.bfill(inplace= True, axis=0)
        flight = flight_T.T
        flight = flight.to_numpy()
        flight, flight_transformed = self.transformation(flight)
        flight = torch.tensor(flight, dtype=torch.float32)
        flight_transformed = torch.tensor(flight_transformed, dtype=torch.float32)

        if self.reverse_masked:
            flight_transformed = torch.flip(flight_transformed, dims=[0])

        if self.reverse_original:
            flight = torch.flip(flight, dims=[0])

    

        pos_pair = (flight.unsqueeze(dim=0), flight_transformed.unsqueeze(dim=0))

        return pos_pair
