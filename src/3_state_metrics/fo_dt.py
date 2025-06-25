#!/usr/bin/env python3
"""
analyze_clusters.py
-------------------

This script takes the output from the LEiDA clustering step and computes
key dynamic metrics: Fractional Occurrence (FO) and Dwell Time (DT).

It generates and saves:
1.  Tidy CSV files for FO and DT for each K.
2.  Grouped boxplots visualizing FO and DT across conditions for each K.

The script uses the same path-building logic as the previous steps for
seamless integration.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# --- PATHING & DATA LOADING ---

def get_clustering_dir(args: argparse.Namespace) -> Path:
    """Constructs the path to the clustering results directory."""
    base_dir = Path(os.getenv("DATA_DIR", "./data"))
    
    if args.data_type == 'source':
        path = base_dir / "clustering_source" / args.cluster_method / args.comparison_type / args.method / f"{args.freq_band}_{args.window_size}"
    elif args.data_type == 'eeglab':
        path = base_dir / "clustering_eeg" / args.cluster_method / args.comparison_type / f"{args.freq_band}_{args.window_size}"
    elif args.data_type == 'beamformer':
        path = base_dir / "clustering_source" / args.cluster_method / args.comparison_type / "beamformer" / f"{args.freq_band}_{args.window_size}"
    else:
        raise ValueError("Invalid data_type specified.")
        
    if not path.exists():
        raise FileNotFoundError(f"Clustering directory not found. Please check your parameters.\nPath: {path.resolve()}")
    return path

# --- METRIC CALCULATION ---

def calculate_fractional_occurrence(labels_dict: Dict, k: int) -> pd.DataFrame:
    """Calculates the fractional occurrence (FO) for each state."""
    fo_results = []
    for condition, subjects_dict in labels_dict.items():
        for subject, labels_array in subjects_dict.items():
            # labels_array is shape (n_epochs, n_windows)
            if labels_array.size == 0: continue
            
            flat_labels = labels_array.flatten()
            counts = np.bincount(flat_labels, minlength=k)
            fractions = counts / len(flat_labels)
            
            for state_idx, fo_value in enumerate(fractions):
                fo_results.append({
                    "condition": condition,
                    "subject": subject,
                    "state": f"State {state_idx}",
                    "FO": fo_value
                })
    return pd.DataFrame(fo_results)


def calculate_dwell_time(labels_dict: Dict, k: int, window_duration_s: float) -> pd.DataFrame:
    """Calculates the mean dwell time (DT) for each state."""
    dt_results = []
    for condition, subjects_dict in labels_dict.items():
        for subject, labels_array in subjects_dict.items():
            # Store all dwell times for this subject, keyed by state
            dwells_by_state = {i: [] for i in range(k)}
            
            # Process each epoch independently
            for epoch_idx in range(labels_array.shape[0]):
                epoch_labels = labels_array[epoch_idx, :]
                if len(epoch_labels) == 0: continue
                
                # Find indices where the state changes
                change_points = np.where(np.diff(epoch_labels) != 0)[0]
                
                # Get the start index of each continuous block
                run_starts = np.concatenate(([0], change_points + 1))
                # Get the state label for each block
                run_states = epoch_labels[run_starts]
                # Get the length of each block
                run_lengths = np.diff(np.concatenate((run_starts, [len(epoch_labels)])))

                for state, length in zip(run_states, run_lengths):
                    dwells_by_state[state].append(length)

            # Now calculate the mean DT for the subject
            for state_idx in range(k):
                all_dwells = dwells_by_state[state_idx]
                mean_dt_windows = np.mean(all_dwells) if all_dwells else 0
                mean_dt_s = mean_dt_windows * window_duration_s
                
                dt_results.append({
                    "condition": condition,
                    "subject": subject,
                    "state": f"State {state_idx}",
                    "DT_s": mean_dt_s
                })
    return pd.DataFrame(dt_results)

# --- VISUALIZATION ---
def plot_metric(
    df: pd.DataFrame,
    col_name: str,
    display_name: str,
    unit: str,
    output_path: Path,
    conditions: List[str],
):
    """
    col_name      : the actual column in `df` (e.g. "FO" or "DT_s")
    display_name  : human label for titles/axes (e.g. "Fractional Occurrence")
    unit          : unit string for the y-axis (e.g. "Proportion" or "Seconds")
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    df['condition'] = pd.Categorical(df['condition'], categories=conditions, ordered=True)

    g = sns.catplot(
        data=df,
        x='state',
        y=col_name,            # <-- use the real column here
        hue='condition',
        kind='box',
        aspect=2.5,
        height=5,
        legend=True,
        palette='viridis'
    )

    # Title & labels use the human string
    g.figure.suptitle(
        f"Distribution of {display_name}",
        y=1.03, fontsize=16
    )
    g.set_axis_labels("Brain State", f"{display_name} ({unit})")

    fn = output_path / f"{col_name.lower()}_distribution.png"
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   -> Saved plot: {fn.name}")


# --- MAIN DRIVER ---

def main():
    parser = argparse.ArgumentParser(
        description="Analyze LEiDA clustering results by computing and visualizing FO and DT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Path-building Arguments (must match clustering script) ---
    parser.add_argument('--cluster_method', type=str, required=True, choices=['kmeans', 'diametrical'])
    parser.add_argument('--data_type', type=str, required=True, choices=['source', 'eeglab', 'beamformer'])
    parser.add_argument('--comparison_type', type=str, default='all')
    parser.add_argument('--method', type=str, default='dSPM')
    parser.add_argument('--freq_band', type=str, default='alpha')
    parser.add_argument('--window_size', type=int, default=256)
    
    # --- Analysis-specific Arguments ---
    parser.add_argument('--fs', type=float, default=256.0, help="Sampling frequency (Hz) used for the data.")
    parser.add_argument('--k_min', type=int, default=2, help="Minimum K to analyze.")
    parser.add_argument('--k_max', type=int, default=20, help="Maximum K to analyze.")
    parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'])
    args = parser.parse_args()

    t0 = time.time()
    clustering_base_dir = get_clustering_dir(args)
    print(f"--- Analysis Started ---")
    print(f"Analyzing results in: {clustering_base_dir.resolve()}")

    window_duration_s = args.window_size / args.fs
    print(f"Window size: {args.window_size} samples | Fs: {args.fs} Hz -> Window duration: {window_duration_s:.3f} s")

    for k in range(args.k_min, args.k_max + 1):
        k_dir = clustering_base_dir / f"k_{k:02d}"
        if not k_dir.exists():
            continue
        
        print(f"\nProcessing K = {k}...")
        
        # Load the clustered labels
        labels_path = k_dir / "labels.pkl"
        with open(labels_path, 'rb') as f:
            labels_dict = pickle.load(f)
        
        # 1. Calculate and save Fractional Occurrence
        fo_df = calculate_fractional_occurrence(labels_dict, k)
        fo_csv_path = k_dir / "fractional_occurrence.csv"
        fo_df.to_csv(fo_csv_path, index=False)
        print(f"   -> Saved FO data: {fo_csv_path.name}")
        
        # 2. Calculate and save Dwell Time
        dt_df = calculate_dwell_time(labels_dict, k, window_duration_s)
        dt_csv_path = k_dir / "dwell_time.csv"
        dt_df.to_csv(dt_csv_path, index=False)
        print(f"   -> Saved DT data: {dt_csv_path.name}")
        
        # 3. Create and save plots
        plot_metric(fo_df,  "FO",   "Fractional Occurrence", "Proportion", k_dir, args.conditions)
        plot_metric(dt_df,  "DT_s", "Dwell Time",           "Seconds",    k_dir, args.conditions)


    run_time = time.time() - t0
    print(f"\n--- Analysis Complete ---")
    print(f"Total execution time: {run_time:.2f} seconds.")
    print(f"Results (CSVs and plots) have been saved inside each 'k_##' subdirectory.")

if __name__ == "__main__":
    main()