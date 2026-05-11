import os
import sys
import pandas as pd

DERVIED_COLS = ['stallindex', 'aoasimple', 'densityratio', 'trueairspeed(ft/min)', 'vspdcalculated']

for subdir in ['test', 'train', 'val']:
    for csv_file in os.listdir(os.path.join(sys.argv[1], subdir)):
        file = os.path.join(sys.argv[1], subdir, csv_file)
        df = pd.read_csv(file)
        df = df.drop(columns=DERVIED_COLS)
        df.to_csv(file)
        print(f'Done with {file}')
