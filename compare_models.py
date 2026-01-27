import pandas as pd
import numpy as np
import json

# Read both CSV files
autoencoder_df = pd.read_csv('autoencoder_per_feature_metrics.csv')
bert_df = pd.read_csv('bert_per_feature_results.csv')

# Merge on feature_index to align the features
merged_df = autoencoder_df.merge(
    bert_df,
    on='feature_index',
    suffixes=('_autoencoder', '_bert')
)

# Calculate differences (negative means BERT is better)
merged_df['mse_diff'] = merged_df['mse_bert'] - merged_df['mse_autoencoder']
merged_df['mae_diff'] = merged_df['mae_bert'] - merged_df['mae_autoencoder']
merged_df['rmse_diff'] = merged_df['rmse_bert'] - merged_df['rmse_autoencoder']

# Calculate average difference across all metrics
merged_df['avg_diff'] = (merged_df['mse_diff'] + merged_df['mae_diff'] + merged_df['rmse_diff']) / 3

# Calculate percentage improvement for each metric
merged_df['mse_pct_improvement'] = ((merged_df['mse_autoencoder'] - merged_df['mse_bert']) / merged_df['mse_autoencoder']) * 100
merged_df['mae_pct_improvement'] = ((merged_df['mae_autoencoder'] - merged_df['mae_bert']) / merged_df['mae_autoencoder']) * 100
merged_df['rmse_pct_improvement'] = ((merged_df['rmse_autoencoder'] - merged_df['rmse_bert']) / merged_df['rmse_autoencoder']) * 100

# Average percentage improvement across all metrics
merged_df['avg_pct_improvement'] = (
    merged_df['mse_pct_improvement'] +
    merged_df['mae_pct_improvement'] +
    merged_df['rmse_pct_improvement']
) / 3

# Sort by average percentage improvement (higher is better for BERT)
merged_df_sorted = merged_df.sort_values('avg_pct_improvement', ascending=False)

# Print overall statistics
print("=" * 80)
print("OVERALL COMPARISON - AVERAGE ACROSS ALL FEATURES")
print("=" * 80)
print(f"Autoencoder - MSE: {autoencoder_df['mse'].mean():.6f}, MAE: {autoencoder_df['mae'].mean():.6f}, RMSE: {autoencoder_df['rmse'].mean():.6f}")
print(f"BERT        - MSE: {bert_df['mse'].mean():.6f}, MAE: {bert_df['mae'].mean():.6f}, RMSE: {bert_df['rmse'].mean():.6f}")
print()

avg_mse_improvement = ((autoencoder_df['mse'].mean() - bert_df['mse'].mean()) / autoencoder_df['mse'].mean()) * 100
avg_mae_improvement = ((autoencoder_df['mae'].mean() - bert_df['mae'].mean()) / autoencoder_df['mae'].mean()) * 100
avg_rmse_improvement = ((autoencoder_df['rmse'].mean() - bert_df['rmse'].mean()) / autoencoder_df['rmse'].mean()) * 100

print(f"Average MSE improvement: {avg_mse_improvement:.2f}%")
print(f"Average MAE improvement: {avg_mae_improvement:.2f}%")
print(f"Average RMSE improvement: {avg_rmse_improvement:.2f}%")
print()

# Features where BERT is better
bert_better_features = merged_df[
    (merged_df['mse_diff'] < 0) &
    (merged_df['mae_diff'] < 0) &
    (merged_df['rmse_diff'] < 0)
]

print("=" * 80)
print(f"BERT OUTPERFORMS AUTOENCODER ON ALL METRICS: {len(bert_better_features)} features")
print("=" * 80)

# Top 10 features where BERT shows the most improvement
print("\nTOP 10 FEATURES WHERE BERT SHOWS MOST IMPROVEMENT:")
print("-" * 80)
for idx, row in merged_df_sorted.head(10).iterrows():
    print(f"\n{row['feature_name_bert']} (index {row['feature_index']}):")
    print(f"  MSE improvement: {row['mse_pct_improvement']:.2f}% ({row['mse_autoencoder']:.4f} -> {row['mse_bert']:.4f})")
    print(f"  MAE improvement: {row['mae_pct_improvement']:.2f}% ({row['mae_autoencoder']:.4f} -> {row['mae_bert']:.4f})")
    print(f"  RMSE improvement: {row['rmse_pct_improvement']:.2f}% ({row['rmse_autoencoder']:.4f} -> {row['rmse_bert']:.4f})")
    print(f"  Avg improvement: {row['avg_pct_improvement']:.2f}%")

# Create results dictionary for JSON output
results = {
    "overall_metrics": {
        "autoencoder": {
            "avg_mse": float(autoencoder_df['mse'].mean()),
            "avg_mae": float(autoencoder_df['mae'].mean()),
            "avg_rmse": float(autoencoder_df['rmse'].mean())
        },
        "bert": {
            "avg_mse": float(bert_df['mse'].mean()),
            "avg_mae": float(bert_df['mae'].mean()),
            "avg_rmse": float(bert_df['rmse'].mean())
        },
        "improvement_percentages": {
            "mse": float(avg_mse_improvement),
            "mae": float(avg_mae_improvement),
            "rmse": float(avg_rmse_improvement)
        }
    },
    "features_where_bert_better_all_metrics": bert_better_features['feature_index'].tolist(),
    "count_bert_better_all_metrics": len(bert_better_features),
    "top_10_features_for_cherry_picking": []
}

# Add top features for cherry picking
for idx, row in merged_df_sorted.head(10).iterrows():
    results["top_10_features_for_cherry_picking"].append({
        "feature_index": int(row['feature_index']),
        "feature_name": row['feature_name_bert'],
        "autoencoder_metrics": {
            "mse": float(row['mse_autoencoder']),
            "mae": float(row['mae_autoencoder']),
            "rmse": float(row['rmse_autoencoder'])
        },
        "bert_metrics": {
            "mse": float(row['mse_bert']),
            "mae": float(row['mae_bert']),
            "rmse": float(row['rmse_bert'])
        },
        "improvements": {
            "mse_pct": float(row['mse_pct_improvement']),
            "mae_pct": float(row['mae_pct_improvement']),
            "rmse_pct": float(row['rmse_pct_improvement']),
            "avg_pct": float(row['avg_pct_improvement'])
        }
    })

# Save to JSON
with open('bert_vs_autoencoder_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save detailed comparison to CSV
comparison_cols = [
    'feature_index', 'feature_name_bert',
    'mse_autoencoder', 'mse_bert', 'mse_pct_improvement',
    'mae_autoencoder', 'mae_bert', 'mae_pct_improvement',
    'rmse_autoencoder', 'rmse_bert', 'rmse_pct_improvement',
    'avg_pct_improvement'
]
merged_df_sorted[comparison_cols].to_csv('detailed_feature_comparison.csv', index=False)

print("\n" + "=" * 80)
print("RESULTS SAVED TO:")
print("  - bert_vs_autoencoder_comparison.json (summary with cherry-picked features)")
print("  - detailed_feature_comparison.csv (full comparison for all features)")
print("=" * 80)
