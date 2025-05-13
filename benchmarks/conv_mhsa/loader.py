import os
import torch
import glob
import random

from db.interface import DBInterface
from benchmarks.conv_mhsa.flight import Flight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATASET_ROOT = '/mnt/crucial/data/ngafid/exports/loci_dataset_fixed_keys_3'
SPLITS = ['test', 'train', 'val']

DATA_PATH = '/home/aidan/nvm_data/NGAFID-LOCI-Data/preprocessed_data/'

def probabilistic_split():
    r = random.random()
    if r < 0.7:
        return 'train'
    elif r < 0.85:
        return 'test'
    else:
        return 'val'

def process_flights(db):
    files = glob.glob(os.path.join(DATASET_ROOT, "*.csv"))

    for file in files:
        fname = os.path.basename(file)
        toks = fname.split('_')
        aircraft_type = str(''.join(toks[:2])) if toks[0].startswith('C') else str(''.join(toks[:1]))
        flight_id = int(toks[-1].split('.')[0])

        data = {
            'id': flight_id,
            'aircraft_type': aircraft_type,
            'filename': file
        }

        db.insert_row('Flight', data)
        print("Inserted", data)

def split_flights(db):
    ids = db.aselect('Flight', 'id', fetch_one=False)
    for fid, in ids:
        split = probabilistic_split()

        data = {
            'id': 1,
            'flight_id': fid,
            'split': split
        }

        db.insert_row('Dataset', data)
        print(split)

class GADataset(Dataset):
    def __init__(self, db: DBInterface, split: str, dataset_id: int | None = None):
        """
        data_tensor: torch.Tensor of shape (N, 4096, 23)
        label_tensor: torch.Tensor of shape (N,) or (N, 1)
        """
        self.db = db
        self.data = []
        self.split = split

        if dataset_id is not None:
            self._load_from_db(dataset_id)
        else:
            self._load_from_disk()

    def _load_from_disk(self):
        path = os.path.join(DATA_PATH, self.split)
        dir_list = os.listdir(path)
        pbar = tqdm(dir_list, desc=f"Processing {len(dir_list)} flights for {self.split}", unit="flight")
        for file in pbar:
            fname = os.path.basename(file)
            toks = fname.split('_')
            aircraft_type = str(''.join(toks[:2])) if toks[0].startswith('C') else str(''.join(toks[:1]))
            flight_id = int(toks[-1].split('.')[0])

            flight = Flight(flight_id, os.path.join(DATA_PATH, self.split, file), aircraft_type)
            flight.process(no_preproc=True)
            self.data.append(flight)

    def _load_from_db(self, dataset_id: int):
        flight_info = self.db.aselectn('Dataset_Contents', ['flight_id', 'aircraft_type', 'filename'], fetch_one=False, split=self.split)
        pbar = tqdm(flight_info, desc=f"Processing {len(flight_info)} flights for {self.split}", unit="flight")
        for fid, acft_type, fname in pbar:
            flight = Flight(fid, fname, acft_type)
            if flight.process():
                self.data.append(flight)

    def __len__(self):
        return len(self.data)

class ClassificationGADataset(GADataset):
    def __init__(self, db: DBInterface, split: str, dataset_id: int | None = None, predict_engines: bool= False):
        """
        data_tensor: torch.Tensor of shape (N, 4096, 23)
        label_tensor: torch.Tensor of shape (N,) or (N, 1)
        """
        super().__init__(db, split, dataset_id)
        self.predict_engines = predict_engines

    def __getitem__(self, idx):
        flight = self.data[idx]
        return flight.get_data(self.predict_engines)

class RegressionGADataset(GADataset):
    def __init__(self,
                 db: DBInterface,
                 split: str,
                 dataset_id: int | None = None,
                 masking_ratio: float = 0.5,
                 mean_mask_length: float = 5,
                 mode: str = 'separate',
                 distribution: str = 'geometric',
                 exclude_feats: [] = None):
        """
        data_tensor: torch.Tensor of shape (N, 4096, 23)
        label_tensor: torch.Tensor of shape (N,) or (N, 1)
        """
        super().__init__(db, split, dataset_id)
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flight = self.data[idx]

        return flight.get_masked_input(self.masking_ratio,
                                       self.mean_mask_length,
                                       self.mode,
                                       self.distribution,
                                       self.exclude_feats)


def main():
    db = DBInterface("sqlite:///benchmarks.db")

    dataset = RegressionGADataset(db, 'test')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, y, f in loader:
        print(f)
        breakpoint()

if __name__ == "__main__":
    main()
