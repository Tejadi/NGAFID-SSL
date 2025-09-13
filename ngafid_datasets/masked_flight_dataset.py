# datasets/masked_flight_dataset.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
def noise_mask(X, masking_ratio, mean_mask_length, mode='separate', distribution='geometric'):
    seq_len, feat_dim = X.shape
    if distribution == 'geometric':
        if mode == 'separate':
            mask = np.ones((seq_len, feat_dim), dtype=bool)
            for m in range(feat_dim):
                mask[:, m] = geom_noise_mask_single(seq_len, mean_mask_length, masking_ratio)
        else:
            mask_seq = geom_noise_mask_single(seq_len, mean_mask_length, masking_ratio)
            mask = np.tile(mask_seq[:, None], (1, feat_dim))
    else:
        if mode == 'separate':
            mask = np.random.rand(seq_len, feat_dim) > masking_ratio  # True = keep, False = mask
        else:
            mask_seq = np.random.rand(seq_len) > masking_ratio
            mask = np.tile(mask_seq[:, None], (1, feat_dim))
    return mask


def geom_noise_mask_single(L, avg_mask_len, masking_ratio):
    mask = np.ones(L, dtype=bool)
    p_m = 1.0 / avg_mask_len                     # prob to end a masked segment
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # prob to end an unmasked segment
    state = False if np.random.rand() < masking_ratio else True  # start in masked state with given ratio
    for i in range(L):
        mask[i] = state  # True = keep original, False = mask out
        if state and np.random.rand() < p_m:
            state = False
        elif (not state) and np.random.rand() < p_u:
            state = True
    return mask


class MaskedFlightDataset(Dataset):
    def __init__(self, flight_paths_df, masking_ratio=0.6, mean_mask_length=3):
        self.paths = flight_paths_df.reset_index(drop=True)
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths.loc[idx, "file_path"]
        flight_df = pd.read_csv(file_path, na_values=[' NaN', 'NaN', 'NaN '])
        # flight = flight_df.iloc[:, :]
        flight = flight_df
        flight = flight.transpose().fillna(method='ffill').fillna(method='bfill').transpose()
        X = flight.to_numpy(dtype=np.float32)  # shape (seq_len, feat_dim)
        mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, mode='separate', distribution='geometric')
        X_masked = X * mask  # 0-out masked positions
        X_tensor = torch.tensor(X_masked, dtype=torch.float32)
        y_tensor = torch.tensor(X, dtype=torch.float32)
        mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
        X_tensor = X_tensor.unsqueeze(0)
        y_tensor = y_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        return X_tensor, y_tensor, mask_tensor
