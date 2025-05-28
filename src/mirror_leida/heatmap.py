#!/usr/bin/env python3
"""
plot_leida_sig_heatmap.py
-------------------------

Visualise `stats_summary.csv` as a grid of heat-maps:
   rows      → cluster ID
   columns   → condition-pair
   colour    → –log10(p_corrected)   (only if significant)
   facets    → separate axis for every K-solution

Non-significant cells are greyed.

Dependencies
------------
numpy, pandas, seaborn, matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LEiDA significance heat-maps")
    p.add_argument("--csv", required=True, type=Path,
                   help="stats_summary.csv produced by leida_occurrence_stats.py")
    p.add_argument("--metric", choices=["p_corrected", "p_raw"], default="p_corrected",
                   help="Which p-value column to colour-code")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold (uses `significant` column if present)")
    p.add_argument("--palette", default="magma",
                   help="Sequential palette name for significant cells")
    p.add_argument("--out", type=Path, default="sig_heatmap.png",
                   help="Output PNG/PDF filename")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _pair_label(row: pd.Series) -> str:
    """Create a consistent string for a condition pair."""
    return f"{row['cond1']} ↔ {row['cond2']}"


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> None:
    args  = _cli()
    df    = pd.read_csv(args.csv)

    if "significant" in df.columns:
        sig_mask = df["significant"].astype(bool).values
    else:              # fall-back: decide by alpha & chosen metric
        sig_mask = df[args.metric].values < args.alpha

    df["pair"]     = df.apply(_pair_label, axis=1)
    df["logp"]     = -np.log10(df[args.metric].clip(lower=1e-300))
    df["logp_sig"] = df["logp"].where(sig_mask, np.nan)   # NaN ⇒ grey mask

    # order columns & rows ----------------------------------------------------
    pair_order = sorted(df["pair"].unique())
    k_values   = sorted(df["k"].unique())

    # global colour range for all facets (makes the palette comparable)
    vmax = np.nanpercentile(df["logp_sig"], 99)   # robust upper bound
    vmin = 0.

    n_facets = len(k_values)
    fig_w    = max(4, 2.5 * n_facets)
    fig, axes = plt.subplots(1, n_facets, figsize=(fig_w, 4),
                             sharey=True,
                             gridspec_kw={"wspace": .15})

    if n_facets == 1:          # unpack so that axes is always iterable
        axes = [axes]

    for ax, k in zip(axes, k_values):
        sub   = df.loc[df["k"] == k]
        pivot = sub.pivot_table(index="cluster",
                                columns="pair",
                                values="logp_sig",
                                aggfunc="first").reindex(columns=pair_order)

        # clusters may be missing in some k (unlikely) → keep numeric order
        pivot = pivot.sort_index()

        # heat-map
        sns.heatmap(pivot,
                    ax=ax,
                    cmap=args.palette,
                    vmin=vmin, vmax=vmax,
                    cbar=(ax is axes[-1]),   # only right-most gets colour-bar
                    linewidths=.5, linecolor="white",
                    mask=pivot.isna(),
                    square=False)

        ax.set_title(f"K = {k}", fontsize=12, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("cluster" if ax is axes[0] else "")
        # rotate pair labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # single colour-bar label
    cax = axes[-1].collections[0].colorbar
    cax.set_label("–log\u2081\u2080(p)", rotation=270, labelpad=12)

    # add a legend patch for “ns”
    grey = mpl.patches.Patch(color="lightgrey", label="n.s.")
    axes[0].legend(handles=[grey], bbox_to_anchor=(1.02, 1.02),
                   loc="upper left", borderaxespad=0.)

    fig.suptitle("Significant cluster differences (after correction)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out.resolve()}")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
