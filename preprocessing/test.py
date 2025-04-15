from pathlib import Path
import pandas as pd

input_path = Path('../../sample_flights')
flight_paths = [f for f in input_path.glob('*.csv')]
cols_set = set(pd.read_csv('default_columns.csv'))
print(cols_set)
print(len(cols_set))

# counter = 0
# for flight_path in flight_paths:
#     flight_data = pd.read_csv(flight_path, na_values=[' NaN', 'NaN', 'NaN '])

#     flight_cols = set(flight_data.columns)
#     mutual = flight_cols & cols_set
#     if len(mutual) != len(cols_set):
#         continue
#     else:
#         counter += 1

# print(counter)

# Load the JSON file into a DataFrame
df = pd.read_json('standard_column_names.json')

# Convert the "columns" key into a dictionary
columns_dict = df["columns"].to_dict()

print(columns_dict)


# // {"columns" : ["AOASimple",
# // "AltAGL",
# // "AltB",
# // "AltGPS",
# // "AltMSL",
# // "AltMSL Lag Diff",
# // "BaroA",
# // "CAS",
# // "COM1",
# // "COM2",
# // "CRS",
# // "Coordination Index",
# // "DensityRatio",
# // "E1 CHT Divergence",
# // "E1 CHT1",
# // "E1 CHT2",
# // "E1 CHT3",
# // "E1 CHT4",
# // "E1 EGT Divergence",
# // "E1 EGT1",
# // "E1 EGT2",
# // "E1 EGT3",
# // "E1 EGT4",
# // "E1 FFlow",
# // "E1 MAP",
# // "E1 OilP",
# // "E1 OilT",
# // "E1 RPM",
# // "E2 CHT1",
# // "E2 EGT Divergence",
# // "E2 EGT1",
# // "E2 EGT2",
# // "E2 EGT3",
# // "E2 EGT4",
# // "E2 FFlow",
# // "E2 MAP",
# // "E2 OilP",
# // "E2 OilT",
# // "E2 RPM",
# // "FQtyL",
# // "FQtyR",
# // "GndSpd",
# // "HAL",
# // "HCDI",
# // "HDG",
# // "HPLfd",
# // "HPLwas",
# // "IAS",
# // "",L"OCI Index",
# // "LatAc",
# // "MagVar",
# // "NormAc",
# // "OAT",
# // "PichC",
# // "Pitch",
# // "Roll",
# // "RollC",
# // "Stall Index",
# // "TAS",
# // "TRK",
# // "Total Fuel",
# // "True Airspeed(ft/min)",
# // "VAL",
# // "VCDI",
# // "VPLwas",
# // "VSpd",
# // "VSpd Calculated",
# // "VSpdG",
# // "WndDr",
# // "WndSpd",
# // "amp1",
# // "amp2",
# // "number_events = 0",
# // "number_events = 1",
# // "number_events = 10",
# // "number_events = 13",
# // "number_events = 2",
# // "number_events = 3",
# // "number_events = 4",
# // "number_events = 5",
# // "number_events = 6",
# // "number_events = 7",
# // "number_events = 8",
# // "number_events = 9",
# // "volt1",
# // "volt2",
# // "#AfcsOn",
# // "#AltAGL"]}