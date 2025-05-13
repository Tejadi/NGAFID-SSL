import torch
import pandas as pd

from typing import Tuple
from datasets.transformation_dataset import mask_transform

INPUT_COLS = ['vspdg',
              'e1egtdivergence',
              'crs',
              'vspdcalculated',
              'trk',
              'normac',
              'altmsl',
              'vspd',
              'oat',
              'hplwas',
              'baroa',
              'e1oilp',
              'ias',
              'latac',
              'e1egt1',
              'densityratio',
              'e1oilt',
              'altmsllagdiff',
              'pitch',
              'tas',
              'fqtyr',
              'totalfuel',
              'trueairspeed(ft/min)',
              'hplfd',
              'magvar',
              'e1egt2',
              'altgps',
              'amp1',
              'fqtyl',
              'volt1',
              'e1fflow',
              'altagl',
              'altb',
              'roll',
              'stallindex',
              'e1egt3',
              'e1rpm',
              'e1egt4',
              'hdg',
              'aoasimple',
              'gndspd']

AIRCRAFT_CLASS = {
    'Cessna172S': 0,
    'PA-44-180': 1,
    'PA-28-181': 2
}

AIRCRAFT_ENGINES = {
    'Cessna172S': 0,
    'PA-28-181': 0,
    'PA-44-180': 1
}

CLASS_AIRCRAFT = {v: k for k, v in AIRCRAFT_CLASS.items()}

class Flight:
    def __init__(self, flight_id: int, filename: str, aircraft_type: str):
        self.flight_id = flight_id
        self.filename = filename
        self.aircraft_type = aircraft_type

    def process(self, no_preproc=False) -> bool:
        df = pd.read_csv(self.filename)

        if no_preproc:
            self.df = df
            return True

        for col in INPUT_COLS:
            if col not in df.columns:
                return False

        df = df[INPUT_COLS]

        n_rows = df.shape[0]
        pad_needed = 4096 - n_rows

        if pad_needed > 0:
            padding_df = pd.DataFrame(0, index=range(pad_needed), columns=df.columns)
            self.df = pd.concat([df, padding_df], ignore_index=True)
        else:
            self.df = df[:4096]

        self.df = self.df.fillna(0)

        return True

    def get_class(self):
        return AIRCRAFT_CLASS[self.aircraft_type]

    def get_engine_class(self):
        return AIRCRAFT_ENGINES[self.aircraft_type]

    def get_data(self, predict_eninges: bool = False) -> Tuple[torch.Tensor, float, int]:
        df_tensor = torch.tensor(self.df.values, dtype=torch.float32)

        acft_class = self.get_engine_class() if predict_eninges else self.get_class()
        acft_class = torch.tensor(acft_class, dtype=torch.long)

        return (df_tensor, acft_class, self.flight_id,)

    def get_masked_input(self,
                         masking_ratio: float = 0.5,
                         mean_mask_length: float = 0.5,
                         mode: str = 'separate',
                         distribution: str = 'geometric',
                         exclude_feats: [] = None):

        flight_data = self.df.values

        flight_data, transformed_flight_data, _ = mask_transform(flight_data,
                                                              masking_ratio=masking_ratio,
                                                              mean_mask_length=mean_mask_length,
                                                              mode=mode,
                                                              distribution=distribution,
                                                              exclude_feats=exclude_feats)

        flight_data = torch.tensor(flight_data, dtype=torch.float32)
        transformed_flight_data = torch.tensor(transformed_flight_data, dtype=torch.float32)

        return transformed_flight_data, flight_data, self.flight_id

