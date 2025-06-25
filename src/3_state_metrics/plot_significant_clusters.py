#!/usr/bin/env python3
"""
visualize_significant_states.py
-------------------------------

This script scans a LEiDA analysis directory for statistically significant
findings in Fractional Occurrence (FO) and generates detailed summary plots
for each significant brain state.

For each state with at least one significant p-value (FDR-corrected), it
creates a two-panel figure:
1.  Left panel: An MNE topomap of the cluster center.
2.  Right panel: A raincloud plot showing the FO distribution across all
    conditions, with significance annotations.

The resulting plots are saved in their respective 'k_##' subdirectories.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection



try:
    import mne
except ImportError:
    print("MNE-Python is required for this script. Please install it with 'pip install mne'")
    mne = None

# --- Plotting Configuration ---
CONDITIONS_ORDER = ['Solo', 'Spontaneous', 'Coordination']
CONDITION_COLORS = {'Solo': '#a3d5ff', 'Spontaneous': '#ffcb8d', 'Coordination': '#ff7886'}

def darken_color(hex_color, factor=0.95):
    """Darken the given color by blending it with black. Factor < 1 darkens the color."""
    rgb = np.array(to_rgb(hex_color))
    dark_rgb = rgb * factor  # scale toward black
    return dark_rgb
# --- Re-usable Plotting Function (as provided) ---

def raincloud(ax: plt.Axes,
              occ: np.ndarray,       # subj × cond
              conds: list[str],
              colors: dict[str,str],
              pair_p: dict[tuple[str,str], tuple[float,bool]],
              title: str):
    """
    Draws violin, box, and scatter plots for occurrence data, annotates
    significant pairs, and includes spaghetti lines connecting subject data.
    """
    n_subj, n_conds = occ.shape
    X = np.arange(n_conds)

    # 1. Pre-compute jitter for scatter plots
    jitter = np.full((n_subj, n_conds), np.nan)
    for ci in range(n_conds):
        mask = ~np.isnan(occ[:, ci])
        jitter[mask, ci] = np.random.uniform(ci - 0.25, ci - 0.15, size=mask.sum())

    # 2. Draw faint spaghetti lines connecting each subject's points
    for si in range(n_subj):
        ys, xs = occ[si, :], jitter[si, :]
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        if valid.sum() > 1:
            ax.plot(xs[valid], ys[valid], color='grey', alpha=0.2, linewidth=0.5, zorder=1)

    # 3. Draw violins, boxplots, and scatter plots
    ymax = np.nanmax(occ) if np.any(~np.isnan(occ)) else 1.0 # Handle all-NaN case
    for i, cond in enumerate(conds):
        data = occ[:, i][~np.isnan(occ[:, i])]
        if len(data) == 0: continue

        # Scatter plot
        subj_idx = np.where(~np.isnan(occ[:, i]))[0]
        ax.scatter(jitter[subj_idx, i], data, s=15, c=colors[cond], alpha=.6, zorder=2)

        # Violin plot
        v = ax.violinplot(data, [i], widths=0.7, showmeans=True, showmedians=False, showextrema=False)
        body = v["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], i, i + 0.4)
        body.set_facecolor(colors[cond]); body.set_alpha(.8); body.set_edgecolor("none")
        mean_color = darken_color(colors[cond], factor=0.9)  # adjust factor as needed
        v["cmeans"].set_color(mean_color)
        # Get the LineCollection representing the mean line
        mean_line = v["cmeans"]
        segments = mean_line.get_segments()

        # Adjust segments: shift and shorten
        new_segments = []
        for seg in segments:
            (x0, y0), (x1, y1) = seg
            
            # Optional: shorten by trimming a portion from both ends (e.g., 20% shorter)
            shorten_factor = 0.7
            x_mid = (x0 + x1) / 2
            half_length = (x1 - x0) * shorten_factor / 2
            new_x0 = x_mid - half_length
            new_x1 = x_mid + half_length
            
            # Optional: shift left
            shift = -0.2  # adjust as needed
            new_x0 += shift
            new_x1 += shift

            new_segments.append([[new_x0, y0], [new_x1, y1]])

        # Update the segments
        mean_line.set_segments(new_segments)

        # v["cmeans"].set_linewidth(2)              # Optional: adjust thickness 


        # Boxplot
        ax.boxplot(data, positions=[i ], widths=0.12, vert=True, showcaps=False, patch_artist=True,
                   boxprops=dict(facecolor=colors[cond], alpha=.7, color='grey'), medianprops=dict(color='grey', alpha=0.7), 
                   whiskerprops=dict(color='grey', alpha=0.7), flierprops=dict(markeredgecolor='grey', markersize=5))

        

    # 4. Add significance annotations
    y_max_plot = ax.get_ylim()[1]
    y_step = (y_max_plot - ymax) * 0.6  # Dynamic step based on plot limits
    y0 = y_max_plot

    sorted_pairs = sorted([p for p, (_, sig) in pair_p.items() if sig], key=lambda p: abs(conds.index(p[0]) - conds.index(p[1])))
    
    for idx, (c1, c2) in enumerate(sorted_pairs):
        p_adj, sig = pair_p[(c1, c2)]
        x1, x2 = conds.index(c1), conds.index(c2)
        y = y0 + idx * y_step*2
        
        ax.plot([x1, x1, x2, x2], [y, y + y_step/4, y + y_step/4, y], c="grey", lw=1.3)
        stars = "***" if p_adj < .001 else "**" if p_adj < .01 else "*"
        ax.text((x1 + x2) / 2, y, stars, ha="center", va="bottom", color="r", fontsize=16)
        ax.text((x1 + x2) / 2, y - y_step, f"p={p_adj:.3g}", ha="center", va="bottom", color="grey", fontsize=7)

    ax.set_xticks(X)
    ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel("Fractional Occurrence", fontsize=11)
    # ax.set_title(title, fontsize=12)
    ax.grid(axis="y", ls="--", alpha=.4)
    ax.autoscale(enable=True, axis='y', tight=False)
    ax.margins(y=0.15) # Add extra space at the top for annotations
    sns.despine(ax=ax, trim=True)


def plot_significant_state(k: int, state_id: int, analysis_dir: Path, mne_info: mne.Info, conditions_order: list[str] = CONDITIONS_ORDER, condition_colors: dict[str, str] = CONDITION_COLORS):
    """
    Generates and saves a composite plot for a single significant state.
    """
    state_name = f"State {state_id}"
    k_dir = analysis_dir / f"k_{k:02d}"
    print(f"  -> Plotting for K={k}, {state_name}...")

    # --- 1. Load Required Data ---
    stats_df = pd.read_csv(analysis_dir / "STATS_GLOBAL_FO_perm.csv")
    fo_df = pd.read_csv(k_dir / "fractional_occurrence.csv")
    centers = np.load(k_dir / "centers.npy")
    center_vec = centers[state_id]

    # --- 2. Prepare Data for Raincloud Plot ---
    # a) Get significance info for all pairs for this state
    state_stats_df = stats_df[(stats_df['K'] == k) & (stats_df['state'] == state_name)]
    pair_p = {}
    for _, row in state_stats_df.iterrows():
        try:
            cond1, cond2 = row.comparison.split('_vs_')
            p_val = row.p_value_fdr_global
            is_sig = row.significant_fdr_global
            # Store with sorted keys to handle C1_vs_C2 and C2_vs_C1 consistently
            pair_p[tuple(sorted((cond1, cond2)))] = (p_val, is_sig)
        except (ValueError, AttributeError):
            continue # Skip if comparison format is unexpected
            
    # Also ensure non-significant pairs are in the dict for completeness
    all_pairs = [(c1, c2) for i, c1 in enumerate(conditions_order) for c2 in conditions_order[i+1:]]
    for c1, c2 in all_pairs:
        key = tuple(sorted((c1, c2)))
        if key not in pair_p:
             pair_p[key] = (1.0, False) # Assume non-significant if not in stats file


    # b) Pivot FO data into a (subjects x conditions) numpy array
    state_fo_df = fo_df[fo_df['state'] == state_name]
    pivot_df = state_fo_df.pivot(index='subject', columns='condition', values='FO')
    # Ensure columns are in the correct, consistent order
    pivot_df = pivot_df.reindex(columns=conditions_order)
    occ_array = pivot_df.to_numpy()

    # --- 3. Create the Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 2]})

    # a) Left Panel: Topomap
    ax_topo = axes[0]
    vmax = np.max(np.abs(center_vec))
    mne.viz.plot_topomap(center_vec, mne_info, axes=ax_topo, show=False, sensors=True, names=mne_info['ch_names'],
                         cmap='bwr', vlim=(-vmax, vmax), contours=0)
    # ax_topo.set_title(f"Center", fontsize=12)

    # b) Right Panel: Raincloud
    ax_rain = axes[1]
    raincloud(
        ax=ax_rain,
        occ=occ_array,
        conds=conditions_order,
        colors=condition_colors,
        pair_p=pair_p,
        title="Fractional Occurrence"
    )

    # --- 4. Final Touches and Save ---
    fig.suptitle(f"K={k}, State {state_id}", fontsize=20, y=0.9, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_png = k_dir / f"state_{state_id}_sig_FO.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"     Saved plot: {output_png.name}")


def main():
    """Main execution function."""
    if mne is None:
        return
        
    parser = argparse.ArgumentParser(
        description="Generate summary plots for statistically significant LEiDA states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        required=True,
        help="Path to the analysis directory containing STATS file and k_## subfolders."
    )
    parser.add_argument(
        "--epochs_file",
        type=Path,
        required=True,
        help="Path to an MNE-readable file (.set, .fif) for channel locations."
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs='+',
        default=CONDITIONS_ORDER,
        help="List of conditions to consider in the analysis."
    )
    parser.add_argument(
        "--condition_colors",
        type=str,
        nargs='+',
        default=list(CONDITION_COLORS.values()),
        help="List of colors corresponding to the conditions."
    )

    args = parser.parse_args()

    conditions_order = args.conditions
    condition_colors = {cond: color for cond, color in zip(conditions_order, args.condition_colors)}

    # --- Validate Paths ---
    stats_file = args.analysis_dir / "STATS_GLOBAL_FO_perm.csv"
    if not args.analysis_dir.is_dir():
        print(f"Error: Analysis directory not found at '{args.analysis_dir}'")
        return
    if not stats_file.exists():
        print(f"Error: Statistics file not found at '{stats_file}'")
        return
    if not args.epochs_file.exists():
        print(f"Error: Epochs file not found at '{args.epochs_file}'")
        return

    # --- Load MNE Info ---
    print(f"Loading channel info from: {args.epochs_file.name}")
    if args.epochs_file.suffix == '.set':
        epochs = mne.io.read_epochs_eeglab(args.epochs_file, verbose=False)
    elif args.epochs_file.suffix == '.fif':
        epochs = mne.read_epochs(args.epochs_file, preload=False, verbose=False)
    else:
        print(f"Error: Unsupported epochs file format '{args.epochs_file.suffix}'")
        return
    epochs.set_montage('biosemi64', verbose=False)  # Set montage if needed
    mne_info = epochs.info

    # --- Find Significant States ---
    print(f"Scanning for significant results in: {stats_file.name}")
    stats_df = pd.read_csv(stats_file)
    significant_rows = stats_df[stats_df['significant_fdr_global'] == True]

    if significant_rows.empty:
        print("No significant results found. Exiting.")
        return

    # Get unique (K, State) pairs that need plotting
    # Extracts the integer from "State 3" -> 3
    significant_rows['state_id'] = significant_rows['state'].str.extract(r'(\d+)').astype(int)
    significant_states = significant_rows[['K', 'state_id']].drop_duplicates().sort_values(by=['K', 'state_id'])
    
    print(f"Found {len(significant_states)} unique states with significant results to plot.")

    # --- Generate Plots for each significant state ---
    for _, row in significant_states.iterrows():
        k, state_id = row['K'], row['state_id']
        k_dir = args.analysis_dir / f"k_{k:02d}"
        if not k_dir.exists():
            print(f"Warning: Directory '{k_dir.name}' not found. Skipping plot for K={k}, State {state_id}.")
            continue
        try:
            plot_significant_state(k, state_id, args.analysis_dir, mne_info, 
                                   conditions_order=conditions_order, condition_colors=condition_colors)
        except Exception as e:
            print(f"Error plotting for K={k}, State {state_id}: {e}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()