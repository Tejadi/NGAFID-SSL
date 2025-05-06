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
    def __init__(self, db: DBInterface, split: str, dataset_id: int = 1):
        """
        data_tensor: torch.Tensor of shape (N, 4096, 23)
        label_tensor: torch.Tensor of shape (N,) or (N, 1)
        """
        self.db = db
        self.data = []
        self.split = split

        self._load_from_db(dataset_id)

    def _load_from_db(self, dataset_id: int):
        flight_info = self.db.aselectn('Dataset_Contents', ['flight_id', 'aircraft_type', 'filename'], fetch_one=False, split=self.split)
        pbar = tqdm(flight_info, desc=f"Processing {len(flight_info)} flights for {self.split}", unit="flight")
        for fid, acft_type, fname in pbar:
            flight = Flight(fid, fname, acft_type)
            if flight.process():
                self.data.append(flight)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flight = self.data[idx]
        return flight.get_data()

def main():
    db = DBInterface("sqlite:///benchmarks.db")

    dataset = GADataset(db, 'test', 1)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, y, f in loader:
        print(f, x.shape, y)

if __name__ == "__main__":
    main()
