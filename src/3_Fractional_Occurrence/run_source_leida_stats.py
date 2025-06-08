#!/usr/bin/env python3
"""
leida_source_stats.py
=====================

Compute state-occurrence probabilities for source-level LEiDA results and
run pair-wise statistics with global multiplicity correction.

Outputs for each K:
  - occurrence.npy: (n_subj x n_cond x k) array of probabilities
  - stats.json: Pairwise statistical test results
  - summary_barplot.png: Bar plot of mean occurrences
  - cluster_C##.png: A combined plot with a brain view and raincloud plot
    for each significant cluster.
"""
from __future__ import annotations
import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

import mne

# --------------------------------------------------------------------------- #
# 1. Command-Line Interface and Helpers
# --------------------------------------------------------------------------- #

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEiDA state-occurrence statistics for source data")
    p.add_argument("--results", required=True, type=Path, help="Folder containing k_## sub-folders from clustering")
    p.add_argument("--source_fif_dir", required=True, type=Path, help="Directory with source-level .fif files (to get ROI names)")
    p.add_argument("--subjects_dir", type=Path, required=True, help="Path to FreeSurfer subjects directory")
    p.add_argument("--parc", type=str, default="aparc", help="Parcellation used (e.g., 'aparc')")
    p.add_argument("--conditions", nargs="+", default=["Coordination", "Spontaneous", "Solo"], help="Condition names")
    p.add_argument("--test", choices=["t", "logit", "wilcoxon"], default="wilcoxon", help="Paired statistical test to use")
    p.add_argument("--corr", choices=["fdr", "bonferroni"], default="fdr", help="Multiplicity correction method")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--dpi", type=int, default=150, help="Resolution for saved plots")
    return p.parse_args()

def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1. - p))

def paired_p(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Return p-value for paired samples, handling NaNs."""
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2: return 1.0
    a, b = a[mask], b[mask]
    if method == "wilcoxon":
        return wilcoxon(a, b, zero_method="zsplit")[1] if np.any(a-b) else 1.0
    if method == "logit":
        a, b = logit(a), logit(b)
    return ttest_rel(a, b)[1]

def collect_subjects(k_dirs: Sequence[Path], conditions: Sequence[str]) -> list[str]:
    """Find all unique subject IDs across all label files."""
    subjects: set[str] = set()
    for kd in k_dirs:
        with (kd / "labels.pkl").open("rb") as f:
            labels = pickle.load(f)
        for c in conditions:
            subjects.update(labels.get(c, {}).keys())
    return sorted(subjects)

def get_roi_names(source_fif_dir: Path) -> list[str]:
    """Get ordered ROI names from a source file."""
    first_file = next(source_fif_dir.glob("*-source-beamformer-epo.fif"), None)
    if not first_file: raise FileNotFoundError(f"No source .fif files in {source_fif_dir}")
    return mne.read_epochs(first_file, verbose='ERROR').ch_names

# --------------------------------------------------------------------------- #
# 2. Plotting Functions (Adapted for Source Space)
# --------------------------------------------------------------------------- #

def brain_view(ax: plt.Axes, center_vec: np.ndarray, all_mne_labels: list[mne.Label],
               ordered_roi_names: list[str], subjects_dir: Path, color: str):
    """
    Renders a top-down (dorsal) view of a brain, highlighting the minority positive ROIs.
    """
    pos_indices = np.where(center_vec > 0)[0]
    pos_roi_names = {ordered_roi_names[i] for i in pos_indices}
    labels_to_plot = [lbl for lbl in all_mne_labels if lbl.name in pos_roi_names]

    brain = mne.viz.Brain("fsaverage", hemi="both", surf="pial", subjects_dir=subjects_dir,
                          background="white", size=(400, 400), cortex='low_contrast')
    brain.show_view('dorsal')
    
    for label in labels_to_plot:
        brain.add_label(label, color=color, alpha=0.85)
    
    ax.imshow(brain.screenshot())
    ax.set_axis_off()
    brain.close()

def raincloud(ax: plt.Axes, occ: np.ndarray, conds: list[str], colors: dict[str, str],
              pair_p: dict[tuple[str, str], tuple[float, bool]], title: str):
    """Draws a raincloud plot to compare occurrence across conditions."""
    n_subj, n_conds = occ.shape
    ymax = 0
    
    for i, cond in enumerate(conds):
        data = occ[:, i][~np.isnan(occ[:, i])]
        if len(data) == 0: continue
        ymax = max(ymax, data.max())
        
        # Violin
        vp = ax.violinplot(data, [i], widths=0.8, showextrema=False)
        vp['bodies'][0].set_facecolor(colors[cond]); vp['bodies'][0].set_alpha(0.5)
        
        # Boxplot
        ax.boxplot(data, positions=[i], widths=0.15, patch_artist=True,
                   showcaps=False, medianprops=dict(color='k'),
                   boxprops=dict(facecolor=colors[cond], alpha=0.7))

        # Jittered scatter
        jitter = np.random.normal(loc=i, scale=0.04, size=len(data))
        ax.scatter(jitter, data, s=15, color='k', alpha=0.6, zorder=2)

    # Significance annotations
    y0, dy = ymax * 1.05, ymax * 0.08
    sig_pairs = [pair for pair, (_, sig) in pair_p.items() if sig]
    for idx, (c1, c2) in enumerate(sig_pairs):
        p_adj, _ = pair_p[(c1,c2)]
        x1, x2 = conds.index(c1), conds.index(c2)
        y = y0 + idx * dy
        ax.plot([x1, x1, x2, x2], [y, y + dy/3, y + dy/3, y], c="k", lw=1.2)
        stars = "***" if p_adj < .001 else "**" if p_adj < .01 else "*"
        ax.text((x1 + x2) / 2, y + dy/3, stars, ha="center", va="bottom", color="r", fontsize=16)

    ax.set_xticks(range(n_conds))
    ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel("Occurrence Probability", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", ls="--", alpha=.4)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, trim=True)

def barplot_all(folder: Path, occ: np.ndarray, conds: Sequence[str], colors: dict[str,str],
                stats_json: dict[str, list[list]], dpi: int):
    """Creates a summary bar plot of mean occurrences across all clusters."""
    k = occ.shape[-1]
    means = np.nanmean(occ, axis=0)
    sem = np.nanstd(occ, axis=0) / np.sqrt(np.sum(~np.isnan(occ), axis=0))

    width = 0.8 / len(conds)
    x = np.arange(k)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * k), 5))

    for ci, cond in enumerate(conds):
        offset = (ci - (len(conds) - 1) / 2) * width
        ax.bar(x + offset, means[ci], width, yerr=sem[ci], capsize=3, color=colors[cond], label=cond)

    # Significance annotations (simplified)
    ymax = (means + sem).max()
    dy = ymax * 0.05
    for c in range(k):
        sig_pairs = [(pair, stats[c][1]) for pair, stats in stats_json.items() if stats[c][2]]
        for idx, (pair, p_adj) in enumerate(sig_pairs):
            c1, c2 = pair.split("__")
            i1, i2 = conds.index(c1), conds.index(c2)
            x1 = x[c] + (i1 - (len(conds) - 1) / 2) * width
            x2 = x[c] + (i2 - (len(conds) - 1) / 2) * width
            y = (means[:, c] + sem[:, c]).max() + dy * (idx + 1)
            ax.plot([x1, x2], [y, y], c="k", lw=1.0)
            stars = "***" if p_adj < .001 else "**" if p_adj < .01 else "*"
            ax.text((x1 + x2) / 2, y, stars, ha="center", va="bottom", color="r", fontsize=12)

    ax.set_xticks(x); ax.set_xticklabels([f"C{ci}" for ci in x], fontsize=9)
    ax.set_ylabel("Mean Occurrence ± SEM"); ax.set_title("State Occurrence Probability")
    ax.legend(frameon=False); ax.grid(axis="y", ls="--", alpha=.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(folder / "summary_barplot.png", dpi=dpi)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. Main Execution
# --------------------------------------------------------------------------- #

def main() -> None:
    args = _cli()
    k_dirs = sorted([d for d in args.results.glob("k_*") if d.is_dir()], key=lambda p: int(p.name.split("_")[1]))
    if not k_dirs:
        print(f"Error: No 'k_*' directories found in {args.results}"); return

    ks = [int(p.name.split("_")[1]) for p in k_dirs]
    subjects = collect_subjects(k_dirs, args.conditions)
    n_subj, n_conds = len(subjects), len(args.conditions)
    cond_colors = dict(zip(args.conditions, sns.color_palette("Set2", n_conds)))
    
    # Load assets needed for brain plotting
    roi_names = get_roi_names(args.source_fif_dir)
    all_mne_labels = mne.read_labels_from_annot("fsaverage", parc=args.parc, subjects_dir=args.subjects_dir)
    all_mne_labels = [lbl for lbl in all_mne_labels if 'unknown' not in lbl.name]

    # --- Pass 1: Calculate all occurrences and raw p-values ---
    all_tests = []
    # ... (this whole section is correct and unchanged) ...
    for k, k_dir in zip(ks, k_dirs):
        with (k_dir / "labels.pkl").open("rb") as f: labels = pickle.load(f)
        occ = np.full((n_subj, n_conds, k), np.nan)
        for ci, cond in enumerate(args.conditions):
            for si, subj in enumerate(subjects):
                if subj in labels.get(cond, {}):
                    flat = labels[cond][subj].ravel()
                    occ[si, ci, :] = np.bincount(flat, minlength=k) / len(flat)
        np.save(k_dir / "occurrence.npy", occ)

        for c in range(k):
            for c1, c2 in itertools.combinations(args.conditions, 2):
                p = paired_p(occ[:, args.conditions.index(c1), c], occ[:, args.conditions.index(c2), c], args.test)
                all_tests.append({'k': k, 'cluster': c, 'cond1': c1, 'cond2': c2, 'p_raw': p})

    # --- Global multiplicity correction ---
    df_tests = pd.DataFrame(all_tests)
    # ... (this section is correct and unchanged) ...
    if args.corr == "fdr":
        df_tests['significant'], df_tests['p_corrected'] = fdrcorrection(df_tests['p_raw'], alpha=args.alpha)
    else: # bonferroni
        reject, p_adj, _, _ = mne.stats.bonferroni_correction(df_tests['p_raw'], alpha=args.alpha)
        df_tests['significant'], df_tests['p_corrected'] = reject, p_adj
    
    df_tests.to_csv(args.results / "stats_summary.csv", index=False)

    # --- Pass 2: Generate plots using corrected p-values ---
    for k, k_dir in zip(ks, k_dirs):
        occ = np.load(k_dir / "occurrence.npy")
        centers = np.load(k_dir / "centers.npy")
        
        # Re-create stats.json for this k from the global dataframe
        stats_json = {}
        for c1, c2 in itertools.combinations(args.conditions, 2):
            key = f"{c1}__{c2}"
            subset = df_tests.query("k == @k and cond1 == @c1 and cond2 == @c2").sort_values("cluster")
            stats_json[key] = subset[['p_raw', 'p_corrected', 'significant']].to_numpy().tolist()

        (k_dir / "stats.json").write_text(json.dumps(stats_json, indent=2))
        
        barplot_all(k_dir, occ, args.conditions, cond_colors, stats_json, args.dpi)

        for c in range(k):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.5]})
            
            center_to_plot = centers[c]
            if np.sum(center_to_plot > 0) > np.sum(center_to_plot < 0):
                center_to_plot *= -1
            
            brain_view(ax1, center_to_plot, all_mne_labels, roi_names, args.subjects_dir, color=f"C{c}")
            
            # =========================================================================
            # THE FIX IS HERE
            # =========================================================================
            # We need to build the `pair_pvals` dictionary correctly.
            pair_pvals = {}
            for pair_str, stats_list in stats_json.items():
                # pair_str is "Coordination__Solo", stats_list is the list of stats for that pair
                c1, c2 = pair_str.split("__")
                p_corrected = stats_list[c][1]  # The corrected p-value is at index 1
                is_significant = stats_list[c][2] # The significance flag is at index 2
                pair_pvals[(c1, c2)] = (p_corrected, is_significant)
            # =========================================================================
            
            raincloud(ax2, occ[:, :, c], args.conditions, cond_colors, pair_pvals, f"Cluster {c} Occurrence")

            fig.tight_layout()
            fig.savefig(k_dir / f"cluster_C{c:02d}.png", dpi=args.dpi)
            plt.close(fig)

    print("\nStatistical analysis complete.")
    sig_df = df_tests.query("significant")
    if not sig_df.empty:
        print("\nSignificant differences after correction:")
        print(sig_df.to_string(index=False))
    else:
        print("\nNo significant differences found after correction.")

if __name__ == "__main__":
    main()