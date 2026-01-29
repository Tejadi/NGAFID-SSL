#!/usr/bin/env python 

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# from ../sample_flights.combine_flight_data import flight_paths
#
import math
import glob
import os

from utils import load_config
from scipy import sparse
from sklearn.preprocessing import normalize

# CSV_FILE = "events.csv"
# FLIGHT_ID_FILE = "flight_ids.csv"
# NUM_FLIGHTS = 7679

config = load_config()

EVENTS = config['paths']['events']
FLIGHTS_PATH = config['paths']['fixed_keys_flights']
FLIGHT_ID_FILE = config['paths']['flight_id_file']


FLIGHTS = glob.glob(os.path.join(FLIGHTS_PATH, "*.csv"))


class ScoreDatasetGenerator:
    def __init__(self):
        self.events = pd.read_csv(EVENTS)
        self.flight_ids = pd.read_csv(FLIGHT_ID_FILE)
        self.scores = None

        self.num_flights = len(self.flight_ids)

        # Cached artifacts for similarity pairing
        self._flight_index = None
        self._tfidf_matrix = None

    def get_scores(self):
        """
        Keeps your existing output (flight_id -> scalar tfidf score) for plotting/debug,
        but also builds a per-flight TF-IDF *vector* matrix for cosine pairing.
        """
        # Count in how many unique flights each event appears (document frequency)
        event_flight_counts = (
            self.events.groupby("name")["flight_id"]
            .nunique()
            .reset_index(name="NUM_FLIGHTS")
        )
        df_map = event_flight_counts.set_index("name")["NUM_FLIGHTS"].to_dict()

        # Term frequency: occurrences of each event per flight
        occurrences = (
            self.events.groupby(["flight_id", "name"])
            .size()
            .reset_index(name="counts")
        )

        # --- Build TF-IDF sparse matrix: rows=flights, cols=events ---
        flight_list = self.flight_ids["flight_id"].astype(int).tolist()
        flight_to_row = {fid: i for i, fid in enumerate(flight_list)}
        self._flight_index = flight_list

        event_names = sorted(occurrences["name"].unique().tolist())
        event_to_col = {evt: j for j, evt in enumerate(event_names)}

        rows = occurrences["flight_id"].map(flight_to_row)
        cols = occurrences["name"].map(event_to_col)
        tf = occurrences["counts"].astype(float)

        # Drop rows for any flight_id not present in FLIGHT_ID_FILE
        valid_mask = rows.notna() & cols.notna()
        rows = rows[valid_mask].astype(int).to_numpy()
        cols = cols[valid_mask].astype(int).to_numpy()
        tf = tf[valid_mask].to_numpy()

        # IDF with your original definition (log10(N / df))
        # Protect against any weird df=0 (shouldn't happen if it appears)
        idf = np.zeros(len(event_names), dtype=float)
        for evt, j in event_to_col.items():
            df = df_map.get(evt, 0)
            if df > 0:
                idf[j] = math.log10(self.num_flights / df)
            else:
                idf[j] = 0.0

        tfidf_data = tf * idf[cols]
        tfidf_matrix = sparse.csr_matrix(
            (tfidf_data, (rows, cols)),
            shape=(len(flight_list), len(event_names)),
            dtype=float,
        )

        self._tfidf_matrix = tfidf_matrix

        # --- Preserve your scalar score output (sum of TF-IDF weights per flight) ---
        tfidf_score = np.asarray(tfidf_matrix.sum(axis=1)).reshape(-1)
        flights_tfidf = pd.DataFrame({"flight_id": flight_list, "tfidf": tfidf_score})

        self.scores = pd.merge(self.flight_ids, flights_tfidf, on="flight_id", how="left")
        self.scores["tfidf"] = self.scores["tfidf"].fillna(0.0)

        self.scores.to_csv("NGAFID-LOCI-Data/flight_safety_scores.csv", index=False)
        return self.scores

    def plot_non_zero_scores(self):
        sns.histplot(self.scores[self.scores["tfidf"] > 0]["tfidf"], kde=True)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title("Non-Zero Score Distribution")
        plt.savefig("Non_Zero_Scores.png")
        plt.close()

    def plot_all_scores(self):
        sns.histplot(self.scores["tfidf"], kde=True)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title("Complete Score Distribution")
        plt.savefig("Scores.png")
        plt.close()

    def pair_generator(self, non_zero=False):
        """
        Generate positive pairs by cosine similarity between per-flight TF-IDF vectors.

        Strategy:
            1) Compute cosine similarity via normalized vectors (dot product).
            2) Greedily match each flight to its most similar *unmatched* flight.
        """
        if self._tfidf_matrix is None or self._flight_index is None:
            # Build tfidf vectors/scores if not already built
            self.get_scores()

        X = self._tfidf_matrix

        # Optionally restrict to flights with at least one non-zero feature
        if non_zero:
            nnz_mask = np.asarray(X.getnnz(axis=1)).reshape(-1) > 0
            idx = np.where(nnz_mask)[0]
            X = X[idx]
            flight_ids = [self._flight_index[i] for i in idx]
        else:
            flight_ids = list(self._flight_index)

        n = len(flight_ids)
        if n < 2:
            return pd.DataFrame({"Positive Pairs": []})

        # Normalize rows so cosine similarity is just dot product
        Xn = normalize(X, norm="l2", axis=1, copy=True)

        # Greedy pairing without reuse
        used = np.zeros(n, dtype=bool)
        pairs = []

        for i in range(n):
            if used[i]:
                continue

            # Similarities to all others
            sims = (Xn[i].dot(Xn.T)).toarray().reshape(-1)

            # Exclude self + already used candidates
            sims[i] = -np.inf
            sims[used] = -np.inf

            j = int(np.argmax(sims))
            if not np.isfinite(sims[j]):
                # No available partner (odd leftover)
                break

            used[i] = True
            used[j] = True
            pairs.append((flight_ids[i], flight_ids[j]))

        return pd.DataFrame({"Positive Pairs": pairs})

s = ScoreDatasetGenerator()
print(s.pair_generator())
s.plot_all_scores()
