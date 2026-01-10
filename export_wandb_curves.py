#!/usr/bin/env python3
"""
Export W&B training curves to PDF for research paper.
"""
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", context="paper", palette="deep")
sns.set_context("paper", font_scale=1.3)

# Initialize W&B API
api = wandb.Api()

# Your run path from the URL: https://wandb.ai/bats-swift/bert-flight-full/runs/j7evn1hb
run = api.run("bats-swift/bert-flight-full/j7evn1hb")

# Select the metrics we want
keys = [
    "epoch/mse",
    "epoch/mae",
    "train/mse_per_position",
    "train/mse_loss",
    "_step"
]

print("Fetching run history...")
# Fetch all history samples (not just summary)
# Use samples parameter to get more data points
hist = run.history(pandas=True, samples=10000)
print(f"Retrieved {len(hist)} rows of data")

# Debug: print available columns and data shape
print(f"Columns: {hist.columns.tolist()}")
print(f"Data shape: {hist.shape}")
print(f"\nFirst few rows:")
print(hist.head(10))

# Filter the data for the specific metrics we need
metrics_to_plot = {
    'epoch/mse': 'validation/mse',
    'epoch/mae': 'validation/mae',
    'train/mse_per_position': 'train/mse_per_position',
    'train/mse_loss': 'train/mse_loss'
}

# Create a PDF with all four plots
output_pdf = "wandb_training_curves.pdf"
with PdfPages(output_pdf) as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=18, fontweight='bold', y=0.995)

    # Use seaborn color palette
    colors = sns.color_palette("husl", 4)
    plot_idx = 0

    for metric_key, display_name in metrics_to_plot.items():
        ax = axes[plot_idx // 2, plot_idx % 2]

        if metric_key in hist.columns:
            # Get non-null values
            data = hist[['_step', metric_key]].dropna()

            if len(data) > 0:
                # Use seaborn lineplot for smoother aesthetics
                sns.lineplot(x='_step', y=metric_key, data=data,
                           ax=ax, linewidth=2.5, marker='o',
                           markersize=6, color=colors[plot_idx],
                           markeredgewidth=0.5, markeredgecolor='white')

                ax.set_xlabel('Step', fontsize=12, fontweight='semibold')
                ax.set_ylabel(display_name.split('/')[-1].upper(),
                            fontsize=12, fontweight='semibold')
                ax.set_title(display_name, fontsize=14, fontweight='bold', pad=10)

                # Improve grid aesthetics
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_axisbelow(True)

                # Add some formatting
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3,3))

                # Add subtle background
                ax.set_facecolor('#f8f9fa')

                print(f"✓ Plotted {metric_key}: {len(data)} data points")
            else:
                print(f"⚠ No data for {metric_key}")
        else:
            print(f"⚠ Metric {metric_key} not found in data")

        plot_idx += 1

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', dpi=600)
    plt.close()

print(f"✓ Saved training curves to {output_pdf}")

# Also save the raw data as CSV for reference
csv_output = "wandb_run_history.csv"
hist.to_csv(csv_output, index=False)
print(f"✓ Saved raw data to {csv_output}")
