#!/usr/bin/env python3
"""
Generate dataset teaser figure for ICML paper.

Creates a multi-panel figure showing:
- Panel A: Time-series traces for 3 flights (one per airframe type)
- Panel B: Missingness heatmap for key features
- Panel C: Safety event timeline
- Panel D: Cross-airframe domain shift (IAS distribution)

Verified readability at \\linewidth in two-column ICML format.

Usage:
    python make_dataset_teaser.py --data_dir /path/to/ngafid
    python make_dataset_teaser.py --data_dir /path/to/ngafid --out figures/teaser.pdf
    python make_dataset_teaser.py --data_dir /path/to/ngafid --figsize 6.5 5.0 --font_size 8
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
np.random.seed(SEED)

# Okabe-Ito colorblind-safe palette (desaturated for print)
OKABE_ITO = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
    'gray': '#666666',
    'lightgray': '#CCCCCC',
}

# Aircraft colors
AIRCRAFT_COLORS = {
    'Cessna 172S': OKABE_ITO['blue'],
    'PA-28-181': OKABE_ITO['orange'],
    'PA-44-180': OKABE_ITO['green'],
}

AIRCRAFT_SHORT = {
    'Cessna 172S': 'C172S',
    'PA-28-181': 'PA-28',
    'PA-44-180': 'PA-44',
}

# Event type short names for legend (readable, not truncated)
EVENT_SHORT_NAMES = {
    'High Altitude Stall': 'High Alt. Stall',
    'Low Airspeed on Climbout': 'Low Spd Climb',
    'VSI on Final': 'VSI on Final',
    'Low Airspeed on Approach': 'Low Spd Appr',
    'Roll': 'Roll',
    'Low Pitch': 'Low Pitch',
    'High Pitch': 'High Pitch',
    'High Altitude Spin': 'High Alt. Spin',
    'Other': 'Other',
}

# Event colors (limit to 5 + Other for clarity)
EVENT_COLORS = {
    'High Altitude Stall': OKABE_ITO['vermillion'],
    'Low Airspeed on Climbout': OKABE_ITO['sky'],
    'VSI on Final': OKABE_ITO['purple'],
    'Low Airspeed on Approach': OKABE_ITO['yellow'],
    'Roll': OKABE_ITO['green'],
    'Other': OKABE_ITO['lightgray'],
}

TOP_EVENT_TYPES = list(EVENT_COLORS.keys())[:-1]  # Exclude 'Other'

# Feature mapping: raw_name -> pretty display name
FEATURE_DISPLAY_NAMES = {
    'AltMSL': 'Altitude',
    'altmsl': 'Altitude',
    'IAS': 'Airspeed',
    'ias': 'Airspeed',
    'Pitch': 'Pitch',
    'pitch': 'Pitch',
    'Roll': 'Roll',
    'roll': 'Roll',
    'HDG': 'Heading',
    'hdg': 'Heading',
    'VSpd': 'Vert Spd',
    'vspd': 'Vert Spd',
    'E1 RPM': 'Eng RPM',
    'e1rpm': 'Eng RPM',
    'E1 FFlow': 'Fuel Flow',
    'e1fflow': 'Fuel Flow',
    'E1 OilP': 'Oil Press',
    'e1oilp': 'Oil Press',
    'E1 OilT': 'Oil Temp',
    'e1oilt': 'Oil Temp',
    'GndSpd': 'Gnd Spd',
    'gndspd': 'Gnd Spd',
    'LatAc': 'Lat Accel',
    'latac': 'Lat Accel',
    'NormAc': 'Norm Accel',
    'normac': 'Norm Accel',
    'AOASimple': 'AoA',
    'aoasimple': 'AoA',
    'BaroA': 'Baro Alt',
    'baroa': 'Baro Alt',
    'Total Fuel': 'Fuel Qty',
    'totalfuel': 'Fuel Qty',
}

# Default missingness features (curated for interpretability, 10 max for readability)
DEFAULT_HEATMAP_FEATURES = [
    ('AltMSL', 'Altitude'),
    ('IAS', 'Airspeed'),
    ('Pitch', 'Pitch'),
    ('Roll', 'Roll'),
    ('HDG', 'Heading'),
    ('VSpd', 'Vert Spd'),
    ('E1 RPM', 'Eng RPM'),
    ('E1 FFlow', 'Fuel Flow'),
    ('GndSpd', 'Gnd Spd'),
    ('AOASimple', 'AoA'),
]


def setup_matplotlib_style(base_font_size=8):
    """Configure matplotlib for ICML-quality output."""
    plt.rcParams.update({
        # Serif font family (CM-like)
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'CMU Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'cm',

        # Font sizes (scaled from base)
        'font.size': base_font_size,
        'axes.titlesize': base_font_size + 1,
        'axes.labelsize': base_font_size,
        'xtick.labelsize': base_font_size - 1,
        'ytick.labelsize': base_font_size - 1.5,  # Slightly smaller for less crowding
        'legend.fontsize': base_font_size - 1.5,
        'figure.titlesize': base_font_size + 2,

        # Line and axis styling
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'xtick.minor.size': 1.0,
        'ytick.minor.size': 1.0,
        'lines.linewidth': 0.9,

        # No grid by default
        'axes.grid': False,

        # PDF settings for vector output with embedded fonts
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Clean background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',

        # Spine visibility
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def normalize_column_name(col):
    """Normalize column name for matching."""
    return col.strip().lower().replace(' ', '').replace('_', '')


def find_column(df, target_names):
    """Find a column matching any target name."""
    if isinstance(target_names, str):
        target_names = [target_names]
    normalized_targets = [normalize_column_name(t) for t in target_names]
    for col in df.columns:
        if normalize_column_name(col) in normalized_targets:
            return col
    return None


def load_flight_data(data_dir, aircraft_type, flight_id, use_preprocessed=False):
    """Load flight data from raw or preprocessed directory."""
    data_path = Path(data_dir)

    if use_preprocessed:
        for split in ['train', 'val', 'test']:
            filepath = data_path / 'preprocessed_data' / split / f"{aircraft_type.replace(' ', '_')}_flight_{flight_id}.csv"
            if filepath.exists():
                return pd.read_csv(filepath), filepath
    else:
        filepath = data_path / 'flights' / f"{aircraft_type.replace(' ', '_')}_flight_{flight_id}.csv"
        if filepath.exists():
            return pd.read_csv(filepath, na_values=[' NaN', 'NaN', 'NaN ', 'nan']), filepath

    return None, None


def load_events(data_dir):
    """Load event annotations."""
    events_path = Path(data_dir) / 'events.csv'
    if events_path.exists():
        return pd.read_csv(events_path)
    return None


def select_representative_flights(data_dir, events_df):
    """Select representative flights (deterministic)."""
    return {
        'Cessna 172S': (504, False),
        'PA-28-181': (2981, False),
        'PA-44-180': (3094, True),
    }


def get_event_color(event_name):
    """Get color for event, grouping rare types as Other."""
    if event_name in EVENT_COLORS:
        return EVENT_COLORS[event_name]
    return EVENT_COLORS['Other']


def add_panel_label(ax, label, x=-0.14, y=1.02):
    """Add bold panel label outside plot area."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='left')


def create_teaser_figure(data_dir, output_path, figsize=(6.5, 5.2),
                         font_size=8, n_features=12, max_events=5,
                         time_window=None, dpi=300):
    """Create the ICML-ready dataset teaser figure."""

    setup_matplotlib_style(font_size)
    data_path = Path(data_dir)

    # Load data
    events_df = load_events(data_dir)
    if events_df is None:
        raise FileNotFoundError(f"events.csv not found in {data_path}")

    flight_selection = select_representative_flights(data_dir, events_df)

    # Load flights
    flight_data = {}
    print("=" * 65)
    print("NGAFID Dataset Teaser Figure Generator (ICML Polish Pass)")
    print("=" * 65)
    print(f"Random seed: {SEED}")
    print(f"Figure size: {figsize[0]:.1f} x {figsize[1]:.1f} inches")
    print(f"Base font size: {font_size} pt")
    print(f"Heatmap features: {n_features}")
    print(f"Max event types: {max_events}")
    print("\nSelected flights:")

    for aircraft_type, (flight_id, use_preprocessed) in flight_selection.items():
        df, filepath = load_flight_data(data_dir, aircraft_type, flight_id, use_preprocessed)
        if df is not None:
            flight_events = events_df[events_df['flight_id'] == flight_id]
            flight_data[aircraft_type] = {
                'df': df,
                'flight_id': flight_id,
                'preprocessed': use_preprocessed,
                'events': flight_events
            }
            source = "preprocessed" if use_preprocessed else "raw"
            missing_pct = df.isnull().sum().sum() / df.size * 100 if not use_preprocessed else 0
            print(f"  {AIRCRAFT_SHORT[aircraft_type]:6s} flight {flight_id:4d}: "
                  f"{len(df):5d} pts, {len(flight_events):2d} events, "
                  f"{missing_pct:.1f}% miss ({source})")

    # ==========================================================================
    # Create figure with careful layout
    # ==========================================================================
    fig = plt.figure(figsize=figsize)

    # GridSpec: 3 rows, 5 columns for fine control
    # Row 0: Time-series (3 panels spanning cols 0-4, last col for legend space)
    # Row 1: Heatmap (cols 0-3) + Domain shift (col 4)
    # Row 2: Event timeline (cols 0-3) + Event legend (col 4)
    gs = GridSpec(3, 6, figure=fig,
                  height_ratios=[1.2, 1.1, 0.65],
                  width_ratios=[0.32, 1, 1, 1, 0.15, 0.8],  # Wider spacer column on left
                  hspace=0.55, wspace=0.4)

    aircraft_order = ['Cessna 172S', 'PA-28-181', 'PA-44-180']

    # ==========================================================================
    # Panel A: Time-series (normalized altitude + airspeed)
    # ==========================================================================
    print("\nHeatmap features used:")

    axes_ts = []
    for i, aircraft_type in enumerate(aircraft_order):
        if aircraft_type not in flight_data:
            continue

        ax = fig.add_subplot(gs[0, i + 1])  # +1 to skip spacer column
        axes_ts.append(ax)
        data = flight_data[aircraft_type]
        df = data['df']

        # Time in minutes
        time = np.arange(len(df)) / 60.0

        # Apply time window if specified
        if time_window:
            mask = time <= time_window
            time = time[mask]
            df = df.iloc[mask]

        # Get and normalize altitude/airspeed
        alt_col = find_column(df, ['AltMSL', 'altmsl'])
        ias_col = find_column(df, ['IAS', 'ias'])

        if alt_col:
            alt = df[alt_col].values
            alt_norm = (alt - np.nanmin(alt)) / (np.nanmax(alt) - np.nanmin(alt) + 1e-6)
            ax.plot(time, alt_norm, '-', color=OKABE_ITO['blue'],
                   linewidth=0.9, alpha=0.85, label='Altitude')

        if ias_col:
            ias = df[ias_col].values
            ias_norm = (ias - np.nanmin(ias)) / (np.nanmax(ias) - np.nanmin(ias) + 1e-6)
            ax.plot(time, ias_norm, '-', color=OKABE_ITO['vermillion'],
                   linewidth=0.9, alpha=0.85, label='Airspeed')

        # Event shading (subtle yellow)
        for _, event in data['events'].iterrows():
            start_t = event['start_line'] / 60.0
            end_t = event['end_line'] / 60.0
            if time_window and start_t > time_window:
                continue
            ax.axvspan(start_t, min(end_t, time[-1]), alpha=0.2,
                      color=OKABE_ITO['yellow'], zorder=0, linewidth=0)

        # Styling
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, time[-1])

        # Reduce x-ticks to 4
        ax.set_xticks(np.linspace(0, time[-1], 4).astype(int))
        ax.tick_params(axis='x', labelbottom=False)  # Labels only on row 1

        # Y-axis only on leftmost
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=font_size)
        else:
            ax.tick_params(axis='y', labelleft=False)

        # Title
        short_name = AIRCRAFT_SHORT[aircraft_type]
        n_events = len(data['events'])
        ax.set_title(f'{short_name} ({n_events} events)', fontsize=font_size + 0.5, pad=3)

        # Legend on first panel only
        if i == 0:
            ax.legend(loc='upper right', framealpha=0.95, edgecolor='none',
                     fontsize=font_size - 2, handlelength=1.2, handletextpad=0.3,
                     borderpad=0.3)

    # ==========================================================================
    # Panel B: Missingness heatmap
    # ==========================================================================
    ax_heat = fig.add_subplot(gs[1, 1:4])  # Skip spacer column

    miss_flight = flight_data.get('Cessna 172S')
    if miss_flight and not miss_flight['preprocessed']:
        df = miss_flight['df']

        # Select features for heatmap
        feature_cols = []
        feature_labels = []
        for raw_name, display_name in DEFAULT_HEATMAP_FEATURES[:n_features]:
            col = find_column(df, [raw_name])
            if col:
                feature_cols.append(col)
                feature_labels.append(display_name)
                print(f"  {raw_name:12s} -> {display_name}")

        if feature_cols:
            # Build missingness matrix
            miss_matrix = df[feature_cols].isnull().values.T

            # Subsample for display (target ~300 columns)
            subsample = max(1, len(df) // 300)
            miss_sub = miss_matrix[:, ::subsample]

            # Custom colormap: white -> vermillion
            cmap = LinearSegmentedColormap.from_list('miss',
                ['#FFFFFF', OKABE_ITO['vermillion']], N=2)

            im = ax_heat.imshow(miss_sub, aspect='auto', cmap=cmap,
                               interpolation='nearest', vmin=0, vmax=1,
                               rasterized=True)  # Rasterize heatmap only

            # Y-axis labels
            ax_heat.set_yticks(np.arange(len(feature_labels)))
            ax_heat.set_yticklabels(feature_labels, fontsize=font_size - 1.5)

            # X-axis: 4 ticks for time
            n_ticks = 4
            tick_locs = np.linspace(0, miss_sub.shape[1] - 1, n_ticks)
            tick_labels = [f"{int(x * subsample / 60)}" for x in tick_locs]
            ax_heat.set_xticks(tick_locs)
            ax_heat.set_xticklabels(tick_labels, fontsize=font_size - 1)
            ax_heat.set_xlabel('Time (min)', fontsize=font_size)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, aspect=10, pad=0.02)
            cbar.set_ticks([0.25, 0.75])
            cbar.ax.set_yticklabels(['OK', 'Miss'], fontsize=font_size - 2)
            cbar.outline.set_linewidth(0.3)

    ax_heat.set_title('Sensor Dropout (C172S)', fontsize=font_size + 0.5, pad=3)

    # Restore spines for heatmap (box around it looks cleaner)
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.3)

    # ==========================================================================
    # Panel D: Domain shift histogram
    # ==========================================================================
    ax_hist = fig.add_subplot(gs[1, 5])  # Adjusted for spacer column

    print("\nDomain shift histogram:")
    for aircraft_type in aircraft_order:
        if aircraft_type not in flight_data:
            continue
        df = flight_data[aircraft_type]['df']
        ias_col = find_column(df, ['IAS', 'ias'])

        if ias_col:
            ias_vals = df[ias_col].dropna().values
            if len(ias_vals) > 2000:
                np.random.seed(SEED)
                ias_vals = np.random.choice(ias_vals, 2000, replace=False)

            short = AIRCRAFT_SHORT[aircraft_type]
            print(f"  {short}: IAS range [{ias_vals.min():.0f}, {ias_vals.max():.0f}] kts")

            ax_hist.hist(ias_vals, bins=30, alpha=0.55, density=True,
                        color=AIRCRAFT_COLORS[aircraft_type], label=short,
                        histtype='stepfilled', linewidth=0.4,
                        edgecolor=AIRCRAFT_COLORS[aircraft_type])

    ax_hist.set_xlabel('IAS (kts)', fontsize=font_size)
    ax_hist.set_ylabel('Density', fontsize=font_size)
    ax_hist.set_title('Domain Shift', fontsize=font_size + 0.5, pad=3)
    ax_hist.legend(fontsize=font_size - 2, loc='upper right', framealpha=0.9,
                  edgecolor='none', handlelength=0.8, handletextpad=0.3,
                  borderpad=0.25, labelspacing=0.2)
    ax_hist.tick_params(labelsize=font_size - 1)

    # Fewer x-ticks
    ax_hist.set_xticks([0, 50, 100, 150])

    # ==========================================================================
    # Panel C: Event timeline
    # ==========================================================================
    ax_ev = fig.add_subplot(gs[2, 1:4])  # Skip spacer column

    y_positions = []
    y_labels = []
    bar_h = 0.5

    print("\nEvent types shown:")
    event_types_used = set()

    for idx, aircraft_type in enumerate(aircraft_order):
        if aircraft_type not in flight_data:
            continue
        data = flight_data[aircraft_type]
        events = data['events']
        flight_len = len(data['df'])

        # Background bar
        ax_ev.barh(idx, 1.0, height=bar_h, color=AIRCRAFT_COLORS[aircraft_type],
                  alpha=0.1, edgecolor='none')

        # Event bars
        for _, ev in events.iterrows():
            start = ev['start_line'] / flight_len
            dur = max((ev['end_line'] - ev['start_line']) / flight_len, 0.005)
            ev_name = ev['name']
            color = get_event_color(ev_name)

            # Track which events we're using
            if ev_name in TOP_EVENT_TYPES[:max_events]:
                event_types_used.add(ev_name)
            else:
                event_types_used.add('Other')

            ax_ev.barh(idx, dur, left=start, height=bar_h,
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.2)

        y_positions.append(idx)
        y_labels.append(AIRCRAFT_SHORT[aircraft_type])

    ax_ev.set_yticks(y_positions)
    ax_ev.set_yticklabels(y_labels, fontsize=font_size - 1)
    ax_ev.set_xlim(0, 1)
    ax_ev.set_ylim(-0.4, len(y_positions) - 0.6)
    ax_ev.set_xlabel('Normalized Flight Time', fontsize=font_size)
    ax_ev.set_title('Safety Events', fontsize=font_size + 0.5, pad=3)
    ax_ev.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # ==========================================================================
    # Event legend (separate panel for clean placement)
    # ==========================================================================
    ax_leg = fig.add_subplot(gs[2, 5])  # Adjusted for spacer column
    ax_leg.axis('off')

    # Build legend handles (only types actually used + Other)
    handles = []
    labels_used = []
    for ev_type in TOP_EVENT_TYPES[:max_events]:
        if ev_type in event_types_used or ev_type == 'Other':
            handles.append(mpatches.Patch(facecolor=EVENT_COLORS[ev_type],
                                         edgecolor='black', linewidth=0.2))
            labels_used.append(EVENT_SHORT_NAMES.get(ev_type, ev_type))
            print(f"  {ev_type}")

    # Add Other if needed
    if 'Other' in event_types_used:
        handles.append(mpatches.Patch(facecolor=EVENT_COLORS['Other'],
                                     edgecolor='black', linewidth=0.2))
        labels_used.append('Other')
        print("  Other")

    ax_leg.legend(handles, labels_used, loc='center left', fontsize=font_size - 2,
                 framealpha=0.95, edgecolor='#CCCCCC', fancybox=False,
                 title='Event Types', title_fontsize=font_size - 1,
                 handlelength=1.0, handletextpad=0.4, borderpad=0.4,
                 labelspacing=0.25)

    # ==========================================================================
    # Final save
    # ==========================================================================
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PDF (vector with rasterized heatmap at high DPI)
    fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight',
               pad_inches=0.02, facecolor='white', edgecolor='none')
    print(f"\nSaved: {output_path}")

    # Save PNG
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight',
               pad_inches=0.02, facecolor='white', edgecolor='none')
    print(f"Saved: {png_path}")

    # Save preview at approximate 2-column width (3.25 inches -> ~315 px at 96 dpi)
    preview_path = output_path.with_name('dataset_teaser_preview.png')
    # Scale to simulate 2-column viewing
    fig.savefig(preview_path, format='png', dpi=int(96 * 3.25 / figsize[0]),
               bbox_inches='tight', pad_inches=0.01, facecolor='white')
    print(f"Saved 2-col preview: {preview_path}")

    plt.close(fig)

    print("=" * 65)
    print("Verified readability at \\linewidth in two-column ICML format.")
    print("=" * 65)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate ICML-ready dataset teaser figure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to NGAFID dataset directory')
    parser.add_argument('--out', type=str, default='figures/dataset_teaser.pdf',
                       help='Output PDF path')
    parser.add_argument('--figsize', type=float, nargs=2, default=[6.5, 5.2],
                       metavar=('W', 'H'), help='Figure size in inches')
    parser.add_argument('--font_size', type=float, default=8,
                       help='Base font size in points')
    parser.add_argument('--n_features_heatmap', type=int, default=10,
                       help='Number of features in missingness heatmap')
    parser.add_argument('--max_event_types', type=int, default=5,
                       help='Max event types to show (rest grouped as Other)')
    parser.add_argument('--time_window', type=float, default=None,
                       help='Optional time window in minutes for time-series')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images')

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    create_teaser_figure(
        args.data_dir,
        args.out,
        figsize=tuple(args.figsize),
        font_size=args.font_size,
        n_features=args.n_features_heatmap,
        max_events=args.max_event_types,
        time_window=args.time_window,
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()
