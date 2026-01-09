#!/usr/bin/env python
"""
Example script to run the TF-IDF retrieval benchmark.

This script demonstrates how to:
1. Initialize the benchmark
2. Run the full evaluation
3. Access and interpret results
"""

from ngafid_datasets.tfidf_retrieval_benchmark import TFIDFRetrievalBenchmark
from pathlib import Path


def main():
    """Run the TF-IDF retrieval benchmark."""

    print("="*80)
    print("TF-IDF FLIGHT RETRIEVAL BENCHMARK")
    print("="*80)
    print()
    print("This benchmark evaluates flight similarity retrieval using TF-IDF vectors")
    print("built from flight event data.")
    print()
    print("Ground Truth: Flights are relevant if they share event types")
    print("Metrics: Precision@k, Recall@k, NDCG@k")
    print("Baselines: Random ranking, Metadata-only (aircraft type)")
    print()

    # Initialize benchmark
    # If you have custom paths, you can pass them here:
    # benchmark = TFIDFRetrievalBenchmark(
    #     events_csv='path/to/events.csv',
    #     flight_ids_csv='path/to/flight_ids.csv',
    #     aircraft_types_csv='path/to/aircraft_types.csv'
    # )
    benchmark = TFIDFRetrievalBenchmark()

    # Run full benchmark with k values
    results = benchmark.run_full_benchmark(
        k_values=[5, 10, 20],
        output_dir='tfidf_benchmark_results'
    )

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print()
    print("Results have been saved to: tfidf_benchmark_results/")
    print("  - results.json: Detailed metrics")
    print("  - benchmark_comparison.png: Visualization")
    print()

    # Example: Access specific results
    print("Example - TF-IDF In-Type Performance:")
    if 'tfidf_in_type' in results:
        tfidf_results = results['tfidf_in_type']
        print(f"  Precision@10: {tfidf_results['P@10']['mean']:.4f} ± {tfidf_results['P@10']['std']:.4f}")
        print(f"  Recall@10:    {tfidf_results['R@10']['mean']:.4f} ± {tfidf_results['R@10']['std']:.4f}")
        print(f"  NDCG@10:      {tfidf_results['NDCG@10']['mean']:.4f} ± {tfidf_results['NDCG@10']['std']:.4f}")
    print()

    # Example: Query a specific flight
    print("Example - Retrieve similar flights for a specific query:")
    if benchmark.valid_flight_ids:
        query_id = benchmark.valid_flight_ids[0]
        print(f"  Query flight ID: {query_id}")

        # Retrieve top-5 similar flights
        similar_flights = benchmark.retrieve(query_id, k=5)

        print(f"  Top-5 most similar flights:")
        for rank, (flight_id, similarity) in enumerate(similar_flights, 1):
            # Check relevance
            relevance = benchmark.compute_relevance(query_id, flight_id, binary=False)
            relevant_str = f"({int(relevance)} shared events)" if relevance > 0 else "(no shared events)"
            print(f"    {rank}. Flight {flight_id}: similarity={similarity:.4f} {relevant_str}")

    print()
    print("="*80)


if __name__ == '__main__':
    main()
