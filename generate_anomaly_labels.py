#!/usr/bin/env python
"""Generate binary anomaly labels from the NGAFID events CSV.

A flight is labelled anomalous (1) if it has at least one event in events.csv,
and normal (0) otherwise.

Usage:
    python generate_anomaly_labels.py \
        --events_csv NGAFID-LOCI-Data/events.csv \
        --flight_dir NGAFID-LOCI-Data/preprocessed_data/test \
        --output anomaly_labels.csv

    Or, if you have a flight_ids CSV instead of a flight directory:
    python generate_anomaly_labels.py \
        --events_csv NGAFID-LOCI-Data/events.csv \
        --flight_ids_csv NGAFID-LOCI-Data/flight_ids.csv \
        --output anomaly_labels.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate binary anomaly labels from NGAFID events")
    parser.add_argument("--events_csv", type=str, required=True,
                        help="Path to NGAFID events.csv")
    parser.add_argument("--flight_dir", type=str, default=None,
                        help="Directory of test flight CSVs (extracts flight IDs from filenames)")
    parser.add_argument("--flight_ids_csv", type=str, default=None,
                        help="CSV with a flight_id column listing all flights")
    parser.add_argument("--output", type=str, default="anomaly_labels.csv",
                        help="Output path (default: anomaly_labels.csv)")
    args = parser.parse_args()

    if args.flight_dir is None and args.flight_ids_csv is None:
        parser.error("Provide either --flight_dir or --flight_ids_csv")

    # Get the full set of flight IDs
    if args.flight_dir is not None:
        flight_dir = Path(args.flight_dir)
        flight_ids = []
        for p in sorted(flight_dir.glob("*.csv")):
            # Expects filenames like flight_00123.csv or flight_123.csv
            fid = int(p.stem.split("flight_")[1])
            flight_ids.append(fid)
        print(f"Found {len(flight_ids)} flights in {args.flight_dir}")
    else:
        df = pd.read_csv(args.flight_ids_csv)
        flight_ids = df["flight_id"].tolist()
        print(f"Found {len(flight_ids)} flights in {args.flight_ids_csv}")

    # Load events and find anomalous flight IDs
    events = pd.read_csv(args.events_csv)
    anomalous_ids = set(events["flight_id"].unique())
    print(f"Found {len(anomalous_ids)} unique flight IDs with events")

    # Build labels
    labels = pd.DataFrame({"flight_id": flight_ids})
    labels["label"] = labels["flight_id"].apply(lambda fid: 1 if fid in anomalous_ids else 0)

    n_pos = labels["label"].sum()
    n_neg = len(labels) - n_pos
    print(f"\nLabel distribution:")
    print(f"  Anomalous (1): {n_pos} ({100*n_pos/len(labels):.1f}%)")
    print(f"  Normal    (0): {n_neg} ({100*n_neg/len(labels):.1f}%)")

    labels.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
