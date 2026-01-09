#!/usr/bin/env python
"""
TF-IDF Retrieval Benchmark for Flight Similarity

This module implements a no-training retrieval benchmark that ranks flights
by TF-IDF cosine similarity and evaluates using shared event types as ground truth.

Task: Given a query flight, retrieve the most similar flights based on TF-IDF vectors
Ground Truth: Flights are relevant if they share event types
Metrics: Precision@k, Recall@k, NDCG@k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json
from tqdm import tqdm

from utils import load_config


class TFIDFRetrievalBenchmark:
    """
    TF-IDF based flight retrieval benchmark.

    Builds TF-IDF vectors from flight events and evaluates retrieval quality
    using shared event types as ground truth relevance.
    """

    def __init__(self, events_csv: Optional[str] = None,
                 flight_ids_csv: Optional[str] = None,
                 aircraft_types_csv: Optional[str] = None):
        """
        Initialize the benchmark.

        Args:
            events_csv: Path to events CSV file
            flight_ids_csv: Path to flight IDs CSV file
            aircraft_types_csv: Path to aircraft types CSV file
        """
        # Load configuration
        config = load_config()

        # Set paths
        self.events_path = events_csv or config['paths']['events']
        self.flight_ids_path = flight_ids_csv or config['paths']['flight_id_file']
        self.aircraft_types_path = aircraft_types_csv or config['paths'].get('aircraft_types')

        # Load data
        print("Loading data...")
        self.events = pd.read_csv(self.events_path)
        self.flight_ids = pd.read_csv(self.flight_ids_path)

        # Load aircraft types if available
        self.aircraft_types = None
        if self.aircraft_types_path and Path(self.aircraft_types_path).exists():
            self.aircraft_types = pd.read_csv(self.aircraft_types_path)
            self.flight_ids = pd.merge(self.flight_ids, self.aircraft_types,
                                      on='flight_id', how='left')

        # Initialize containers
        self.tfidf_matrix = None
        self.vectorizer = None
        self.flight_to_idx = {}
        self.idx_to_flight = {}
        self.event_sets = {}  # Maps flight_id to set of event types
        self.flight_documents = {}  # Maps flight_id to text document

        # Results storage
        self.results = {}

    def prepare_data(self):
        """
        Prepare flight documents and event sets for retrieval.

        Creates:
        - Text documents from events for each flight (for TF-IDF)
        - Sets of event types per flight (for relevance judgment)
        """
        print("Preparing flight documents and event sets...")

        # Group events by flight
        flight_events = self.events.groupby('flight_id')['name'].apply(list).to_dict()

        # Create documents and event sets
        valid_flights = []
        for flight_id in tqdm(self.flight_ids['flight_id'], desc="Processing flights"):
            if flight_id in flight_events:
                events = flight_events[flight_id]
                # Document: space-separated event names (with repetition for TF)
                self.flight_documents[flight_id] = ' '.join(events)
                # Event set: unique event types for this flight
                self.event_sets[flight_id] = set(events)
                valid_flights.append(flight_id)

        # Filter to flights with events
        self.valid_flight_ids = valid_flights
        print(f"Found {len(self.valid_flight_ids)} flights with events")

        # Create flight ID mappings
        self.flight_to_idx = {fid: idx for idx, fid in enumerate(self.valid_flight_ids)}
        self.idx_to_flight = {idx: fid for fid, idx in self.flight_to_idx.items()}

    def build_tfidf_vectors(self):
        """
        Build TF-IDF vector representations for all flights.

        Uses sklearn's TfidfVectorizer with:
        - Term frequency (TF): count of each event type in flight
        - Inverse document frequency (IDF): log(N / df) where df is flights with event
        """
        print("Building TF-IDF vectors...")

        # Create ordered list of documents
        documents = [self.flight_documents[fid] for fid in self.valid_flight_ids]

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\b\w+\b',  # Match individual event names
            lowercase=False,  # Event names are already normalized
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False  # Use raw term frequency
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Normalize for cosine similarity (should already be normalized, but ensure)
        self.tfidf_matrix = normalize(self.tfidf_matrix, norm='l2', axis=1)

        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def compute_relevance(self, query_id: int, candidate_id: int,
                         binary: bool = True) -> float:
        """
        Compute relevance between query and candidate flight.

        Args:
            query_id: Query flight ID
            candidate_id: Candidate flight ID
            binary: If True, return 1 if any shared events, else 0
                   If False, return count of shared events

        Returns:
            Relevance score
        """
        query_events = self.event_sets.get(query_id, set())
        candidate_events = self.event_sets.get(candidate_id, set())

        shared_events = query_events & candidate_events

        if binary:
            return 1.0 if len(shared_events) > 0 else 0.0
        else:
            return float(len(shared_events))

    def retrieve(self, query_id: int, k: int = 100,
                exclude_same_aircraft: bool = False) -> List[Tuple[int, float]]:
        """
        Retrieve top-k most similar flights for a query.

        Args:
            query_id: Query flight ID
            k: Number of results to return
            exclude_same_aircraft: If True, only return flights with different aircraft type

        Returns:
            List of (flight_id, similarity_score) tuples, sorted by score descending
        """
        if query_id not in self.flight_to_idx:
            return []

        query_idx = self.flight_to_idx[query_id]
        query_vector = self.tfidf_matrix[query_idx]

        # Compute similarities to all flights
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get aircraft type filtering if needed
        if exclude_same_aircraft and self.aircraft_types is not None:
            query_aircraft = self.flight_ids[
                self.flight_ids['flight_id'] == query_id
            ]['aircraft_type'].values

            if len(query_aircraft) > 0:
                query_aircraft = query_aircraft[0]
                # Mask out same aircraft type
                for idx, fid in self.idx_to_flight.items():
                    candidate_aircraft = self.flight_ids[
                        self.flight_ids['flight_id'] == fid
                    ]['aircraft_type'].values
                    if len(candidate_aircraft) > 0 and candidate_aircraft[0] == query_aircraft:
                        similarities[idx] = -np.inf

        # Exclude the query itself
        similarities[query_idx] = -np.inf

        # Get top-k indices
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        # Convert to flight IDs and scores
        results = [
            (self.idx_to_flight[idx], similarities[idx])
            for idx in top_k_indices if similarities[idx] > -np.inf
        ]

        return results[:k]

    def random_baseline(self, query_id: int, k: int = 100,
                       exclude_same_aircraft: bool = False) -> List[Tuple[int, float]]:
        """
        Random ranking baseline.

        Args:
            query_id: Query flight ID
            k: Number of results to return
            exclude_same_aircraft: If True, only return flights with different aircraft type

        Returns:
            List of (flight_id, random_score) tuples
        """
        candidates = [fid for fid in self.valid_flight_ids if fid != query_id]

        if exclude_same_aircraft and self.aircraft_types is not None:
            query_aircraft = self.flight_ids[
                self.flight_ids['flight_id'] == query_id
            ]['aircraft_type'].values

            if len(query_aircraft) > 0:
                query_aircraft = query_aircraft[0]
                candidates = [
                    fid for fid in candidates
                    if self.flight_ids[self.flight_ids['flight_id'] == fid]['aircraft_type'].values[0] != query_aircraft
                ]

        # Random scores
        np.random.shuffle(candidates)
        scores = np.random.rand(len(candidates))

        results = list(zip(candidates, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def metadata_baseline(self, query_id: int, k: int = 100,
                         exclude_same_aircraft: bool = False) -> List[Tuple[int, float]]:
        """
        Metadata-only baseline (rank by same aircraft type).

        Args:
            query_id: Query flight ID
            k: Number of results to return
            exclude_same_aircraft: If True, only return flights with different aircraft type

        Returns:
            List of (flight_id, metadata_score) tuples
        """
        if self.aircraft_types is None:
            # Fall back to random if no metadata
            return self.random_baseline(query_id, k, exclude_same_aircraft)

        query_aircraft = self.flight_ids[
            self.flight_ids['flight_id'] == query_id
        ]['aircraft_type'].values

        if len(query_aircraft) == 0:
            return self.random_baseline(query_id, k, exclude_same_aircraft)

        query_aircraft = query_aircraft[0]

        candidates = []
        for fid in self.valid_flight_ids:
            if fid == query_id:
                continue

            candidate_aircraft = self.flight_ids[
                self.flight_ids['flight_id'] == fid
            ]['aircraft_type'].values

            if len(candidate_aircraft) == 0:
                continue

            candidate_aircraft = candidate_aircraft[0]

            if exclude_same_aircraft and candidate_aircraft == query_aircraft:
                continue

            # Score: 1.0 if same aircraft, 0.0 otherwise
            score = 1.0 if candidate_aircraft == query_aircraft else 0.0
            candidates.append((fid, score))

        # Shuffle within same score to break ties randomly
        np.random.shuffle(candidates)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:k]

    def precision_at_k(self, query_id: int, retrieved: List[Tuple[int, float]],
                      k: int, binary: bool = True) -> float:
        """
        Compute Precision@k.

        Args:
            query_id: Query flight ID
            retrieved: List of (flight_id, score) retrieved results
            k: Cutoff position
            binary: Use binary or weighted relevance

        Returns:
            Precision@k score
        """
        if len(retrieved) == 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_count = sum(
            1 for fid, _ in top_k
            if self.compute_relevance(query_id, fid, binary=binary) > 0
        )

        return relevant_count / k

    def recall_at_k(self, query_id: int, retrieved: List[Tuple[int, float]],
                   k: int, binary: bool = True) -> float:
        """
        Compute Recall@k.

        Args:
            query_id: Query flight ID
            retrieved: List of (flight_id, score) retrieved results
            k: Cutoff position
            binary: Use binary or weighted relevance

        Returns:
            Recall@k score
        """
        # Get all relevant flights
        all_relevant = [
            fid for fid in self.valid_flight_ids
            if fid != query_id and self.compute_relevance(query_id, fid, binary=binary) > 0
        ]

        if len(all_relevant) == 0:
            return 0.0  # No relevant items

        top_k = retrieved[:k]
        relevant_retrieved = sum(
            1 for fid, _ in top_k
            if self.compute_relevance(query_id, fid, binary=binary) > 0
        )

        return relevant_retrieved / len(all_relevant)

    def ndcg_at_k(self, query_id: int, retrieved: List[Tuple[int, float]],
                 k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@k.

        Args:
            query_id: Query flight ID
            retrieved: List of (flight_id, score) retrieved results
            k: Cutoff position

        Returns:
            NDCG@k score
        """
        # Get relevance scores (weighted by number of shared events)
        top_k = retrieved[:k]
        relevances = [
            self.compute_relevance(query_id, fid, binary=False)
            for fid, _ in top_k
        ]

        # Compute DCG
        dcg = relevances[0] if len(relevances) > 0 else 0.0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 1)

        # Compute ideal DCG
        all_relevances = [
            self.compute_relevance(query_id, fid, binary=False)
            for fid in self.valid_flight_ids if fid != query_id
        ]
        all_relevances.sort(reverse=True)

        idcg = all_relevances[0] if len(all_relevances) > 0 else 0.0
        for i in range(1, min(k, len(all_relevances))):
            idcg += all_relevances[i] / np.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_queries(self, query_ids: Optional[List[int]] = None,
                        k_values: List[int] = [5, 10, 20],
                        cross_type: bool = False,
                        method: str = 'tfidf') -> Dict:
        """
        Evaluate retrieval performance across multiple queries.

        Args:
            query_ids: List of query flight IDs (if None, use all valid flights)
            k_values: List of k values for metrics
            cross_type: If True, only retrieve different aircraft types
            method: Retrieval method ('tfidf', 'random', 'metadata')

        Returns:
            Dictionary of results with mean and std for each metric
        """
        if query_ids is None:
            # Use all flights with at least one relevant item
            query_ids = [
                fid for fid in self.valid_flight_ids
                if sum(1 for other in self.valid_flight_ids
                      if other != fid and self.compute_relevance(fid, other) > 0) > 0
            ]

        print(f"Evaluating {method} on {len(query_ids)} queries...")

        # Storage for metrics
        metrics = defaultdict(list)

        # Retrieve method
        if method == 'tfidf':
            retrieve_fn = self.retrieve
        elif method == 'random':
            retrieve_fn = self.random_baseline
        elif method == 'metadata':
            retrieve_fn = self.metadata_baseline
        else:
            raise ValueError(f"Unknown method: {method}")

        # Evaluate each query
        for query_id in tqdm(query_ids, desc=f"Evaluating {method}"):
            # Retrieve results
            max_k = max(k_values)
            retrieved = retrieve_fn(query_id, k=max_k, exclude_same_aircraft=cross_type)

            # Compute metrics for each k
            for k in k_values:
                p_at_k = self.precision_at_k(query_id, retrieved, k)
                r_at_k = self.recall_at_k(query_id, retrieved, k)
                ndcg_k = self.ndcg_at_k(query_id, retrieved, k)

                metrics[f'P@{k}'].append(p_at_k)
                metrics[f'R@{k}'].append(r_at_k)
                metrics[f'NDCG@{k}'].append(ndcg_k)

        # Compute statistics
        results = {}
        for metric_name, values in metrics.items():
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

        return results

    def run_full_benchmark(self, k_values: List[int] = [5, 10, 20],
                          output_dir: str = 'tfidf_benchmark_results') -> Dict:
        """
        Run complete benchmark with all methods and settings.

        Args:
            k_values: List of k values for metrics
            output_dir: Directory to save results

        Returns:
            Dictionary containing all results
        """
        print("\n" + "="*60)
        print("TF-IDF RETRIEVAL BENCHMARK")
        print("="*60)

        # Prepare data
        self.prepare_data()
        self.build_tfidf_vectors()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Run evaluations
        all_results = {}

        # 1. In-type retrieval (standard)
        print("\n--- In-Type Retrieval ---")
        for method in ['tfidf', 'random', 'metadata']:
            results = self.evaluate_queries(k_values=k_values, cross_type=False, method=method)
            all_results[f'{method}_in_type'] = results

        # 2. Cross-type retrieval (different aircraft)
        if self.aircraft_types is not None:
            print("\n--- Cross-Type Retrieval ---")
            for method in ['tfidf', 'random', 'metadata']:
                results = self.evaluate_queries(k_values=k_values, cross_type=True, method=method)
                all_results[f'{method}_cross_type'] = results

        # Save results
        self.results = all_results
        self.save_results(output_path)

        # Generate table
        self.print_results_table(k_values)

        # Generate plots
        self.plot_results(k_values, output_path)

        print(f"\nResults saved to {output_path}")

        return all_results

    def save_results(self, output_dir: Path):
        """Save results to JSON file."""
        # Prepare serializable results
        serializable_results = {}
        for key, metrics in self.results.items():
            serializable_results[key] = {
                metric_name: {
                    'mean': float(stats['mean']),
                    'std': float(stats['std'])
                }
                for metric_name, stats in metrics.items()
            }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def print_results_table(self, k_values: List[int]):
        """Print results as a formatted table."""
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)

        # Determine which settings we have
        has_cross_type = any('cross_type' in key for key in self.results.keys())

        # Table header
        header = "Method".ljust(20)
        for k in k_values:
            header += f"P@{k}".ljust(12) + f"R@{k}".ljust(12) + f"NDCG@{k}".ljust(12)
        print(header)
        print("-" * len(header))

        # Print results
        for method in ['tfidf', 'random', 'metadata']:
            # In-type results
            key = f'{method}_in_type'
            if key in self.results:
                row = f"{method} (in-type)".ljust(20)
                for k in k_values:
                    p = self.results[key][f'P@{k}']['mean']
                    r = self.results[key][f'R@{k}']['mean']
                    n = self.results[key][f'NDCG@{k}']['mean']
                    row += f"{p:.4f}".ljust(12) + f"{r:.4f}".ljust(12) + f"{n:.4f}".ljust(12)
                print(row)

            # Cross-type results
            if has_cross_type:
                key = f'{method}_cross_type'
                if key in self.results:
                    row = f"{method} (cross-type)".ljust(20)
                    for k in k_values:
                        p = self.results[key][f'P@{k}']['mean']
                        r = self.results[key][f'R@{k}']['mean']
                        n = self.results[key][f'NDCG@{k}']['mean']
                        row += f"{p:.4f}".ljust(12) + f"{r:.4f}".ljust(12) + f"{n:.4f}".ljust(12)
                    print(row)

        print("="*80)

    def plot_results(self, k_values: List[int], output_dir: Path):
        """Generate visualization of results."""
        sns.set_style("whitegrid")

        # Determine which settings we have
        has_cross_type = any('cross_type' in key for key in self.results.keys())

        # Create figure
        n_plots = 2 if has_cross_type else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        # Plot 1: In-type P@10
        ax = axes[0]
        methods = ['TF-IDF', 'Random', 'Metadata']
        method_keys = ['tfidf_in_type', 'random_in_type', 'metadata_in_type']

        p10_means = []
        p10_stds = []
        for key in method_keys:
            if key in self.results:
                p10_means.append(self.results[key]['P@10']['mean'])
                p10_stds.append(self.results[key]['P@10']['std'])
            else:
                p10_means.append(0)
                p10_stds.append(0)

        x = np.arange(len(methods))
        bars = ax.bar(x, p10_means, yerr=p10_stds, capsize=5,
                     color=['#2E86C1', '#95A5A6', '#E67E22'])
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Precision@10', fontsize=12)
        ax.set_title('In-Type Retrieval Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # Plot 2: Cross-type P@10 (if available)
        if has_cross_type:
            ax = axes[1]
            method_keys = ['tfidf_cross_type', 'random_cross_type', 'metadata_cross_type']

            p10_means = []
            p10_stds = []
            for key in method_keys:
                if key in self.results:
                    p10_means.append(self.results[key]['P@10']['mean'])
                    p10_stds.append(self.results[key]['P@10']['std'])
                else:
                    p10_means.append(0)
                    p10_stds.append(0)

            bars = ax.bar(x, p10_means, yerr=p10_stds, capsize=5,
                         color=['#2E86C1', '#95A5A6', '#E67E22'])
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel('Precision@10', fontsize=12)
            ax.set_title('Cross-Type Retrieval Performance', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.set_ylim(0, 1.0)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {output_dir / 'benchmark_comparison.png'}")


def main():
    """Run the benchmark."""
    benchmark = TFIDFRetrievalBenchmark()
    results = benchmark.run_full_benchmark(k_values=[5, 10, 20])


if __name__ == '__main__':
    main()
