#!/usr/bin/env python3
"""
run_statistics_global_fdr.py
----------------------------

Performs non-parametric statistical testing on dynamic metrics (FO, DT)
with a global FDR correction applied across all tested values of K.

This provides a more rigorous control for multiple comparisons when the
choice of K is exploratory.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from itertools import combinations
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection

# --- PATHING HELPER (Identical to previous script) ---

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

# --- CORE ANALYSIS FUNCTION ---

def perform_all_tests(clustering_base_dir: Path, k_min: int, k_max: int, conditions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loops through all K, loads data, and performs Wilcoxon tests,
    returning two DataFrames (for FO and DT) with raw p-values.
    """
    all_fo_results = []
    all_dt_results = []

    for k in range(k_min, k_max + 1):
        k_dir = clustering_base_dir / f"k_{k:02d}"
        if not k_dir.exists(): continue

        print(f"  - Calculating raw p-values for K={k}...")
        
        # --- Process FO ---
        fo_df = pd.read_csv(k_dir / "fractional_occurrence.csv")
        for cond1, cond2 in combinations(conditions, 2):
            for state_name in sorted(fo_df['state'].unique()):
                data1 = fo_df[(fo_df['condition'] == cond1) & (fo_df['state'] == state_name)]
                data2 = fo_df[(fo_df['condition'] == cond2) & (fo_df['state'] == state_name)]
                merged = pd.merge(data1, data2, on='subject')
                
                stat, p_raw = (np.nan, np.nan)
                if merged.shape[0] >= 5:
                    diff = merged['FO_x'] - merged['FO_y']
                    if not np.all(diff == 0):
                        stat, p_raw = wilcoxon(diff)
                
                all_fo_results.append({
                    'K': k, 'state': state_name, 'comparison': f"{cond1}_vs_{cond2}",
                    'metric': 'FO', 'statistic': stat, 'p_value_raw': p_raw
                })

        # --- Process DT ---
        dt_df = pd.read_csv(k_dir / "dwell_time.csv")
        for cond1, cond2 in combinations(conditions, 2):
            for state_name in sorted(dt_df['state'].unique()):
                data1 = dt_df[(dt_df['condition'] == cond1) & (dt_df['state'] == state_name)]
                data2 = dt_df[(dt_df['condition'] == cond2) & (dt_df['state'] == state_name)]
                merged = pd.merge(data1, data2, on='subject')

                stat, p_raw = (np.nan, np.nan)
                if merged.shape[0] >= 5:
                    diff = merged['DT_s_x'] - merged['DT_s_y']
                    if not np.all(diff == 0):
                        stat, p_raw = wilcoxon(diff)

                all_dt_results.append({
                    'K': k, 'state': state_name, 'comparison': f"{cond1}_vs_{cond2}",
                    'metric': 'DT_s', 'statistic': stat, 'p_value_raw': p_raw
                })
                
    return pd.DataFrame(all_fo_results), pd.DataFrame(all_dt_results)

def apply_global_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Applies FDR correction to a dataframe of raw p-values."""
    if df.empty:
        return df
        
    # Drop rows where p-value couldn't be calculated
    valid_p_rows = df.dropna(subset=['p_value_raw'])
    
    if valid_p_rows.empty:
        df['p_value_fdr_global'] = np.nan
        df['significant_fdr_global'] = False
        return df

    # Apply FDR correction on the valid p-values
    p_values_raw = valid_p_rows['p_value_raw'].values
    rejected, p_values_corrected = fdrcorrection(p_values_raw, alpha=0.05, method='indep')
    
    # Create a new column in the original DataFrame
    df['p_value_fdr_global'] = np.nan
    df['significant_fdr_global'] = False
    
    # Map the corrected values back using the index
    df.loc[valid_p_rows.index, 'p_value_fdr_global'] = p_values_corrected
    df.loc[valid_p_rows.index, 'significant_fdr_global'] = rejected
    
    return df

# --- MAIN DRIVER ---
def main():
    parser = argparse.ArgumentParser(
        description="Run non-parametric statistics with GLOBAL FDR correction across all K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Path-building Arguments (must match previous scripts) ---
    parser.add_argument('--cluster_method', type=str, required=True, choices=['kmeans', 'diametrical'])
    parser.add_argument('--data_type', type=str, required=True, choices=['source', 'eeglab', 'beamformer'])
    parser.add_argument('--comparison_type', type=str, default='all')
    parser.add_argument('--method', type=str, default='dSPM')
    parser.add_argument('--freq_band', type=str, default='alpha')
    parser.add_argument('--window_size', type=int, default=256)
    
    # --- Analysis-specific Arguments ---
    parser.add_argument('--k_min', type=int, default=2, help="Minimum K to analyze.")
    parser.add_argument('--k_max', type=int, default=20, help="Maximum K to analyze.")
    parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'])
    args = parser.parse_args()

    if len(args.conditions) < 2:
        raise ValueError("At least two conditions are required for statistical comparison.")

    t0 = time.time()
    clustering_base_dir = get_clustering_dir(args)
    print("--- Global Statistical Analysis Started ---")
    print(f"Reading from: {clustering_base_dir.resolve()}")
    
    # 1. Gather all raw p-values from all K's
    print("\nStep 1: Gathering raw p-values from all K directories...")
    fo_results_raw, dt_results_raw = perform_all_tests(
        clustering_base_dir, args.k_min, args.k_max, args.conditions
    )
    
    # 2. Apply global FDR correction
    print("\nStep 2: Applying global FDR correction across all tests...")
    fo_results_corrected = apply_global_fdr(fo_results_raw)
    dt_results_corrected = apply_global_fdr(dt_results_raw)

    total_fo_tests = len(fo_results_corrected.dropna(subset=['p_value_raw']))
    total_dt_tests = len(dt_results_corrected.dropna(subset=['p_value_raw']))
    print(f"  - Total FO tests corrected: {total_fo_tests}")
    print(f"  - Total DT tests corrected: {total_dt_tests}")

    # 3. Save the final, comprehensive result files
    print("\nStep 3: Saving comprehensive results...")
    
    fo_output_path = clustering_base_dir / "STATS_GLOBAL_FO.csv"
    fo_results_corrected.to_csv(fo_output_path, index=False, float_format='%.5f')
    print(f"  -> Saved FO results to: {fo_output_path}")

    dt_output_path = clustering_base_dir / "STATS_GLOBAL_DT.csv"
    dt_results_corrected.to_csv(dt_output_path, index=False, float_format='%.5f')
    print(f"  -> Saved DT results to: {dt_output_path}")
    
    run_time = time.time() - t0
    print(f"\n--- Analysis Complete in {run_time:.2f} seconds ---")
    
    # Report significant findings
    sig_fo = fo_results_corrected[fo_results_corrected['significant_fdr_global'] == True]
    sig_dt = dt_results_corrected[dt_results_corrected['significant_fdr_global'] == True]

    print("\n--- Summary of Significant Findings (p_fdr_global < 0.05) ---")
    if not sig_fo.empty:
        print("\nFractional Occurrence (FO):")
        print(sig_fo.to_string(index=False))
    else:
        print("\nFractional Occurrence (FO): No significant results found.")

    if not sig_dt.empty:
        print("\nDwell Time (DT):")
        print(sig_dt.to_string(index=False))
    else:
        print("\nDwell Time (DT): No significant results found.")

if __name__ == "__main__":
    main()