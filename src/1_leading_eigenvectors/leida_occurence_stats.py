#!/usr/bin/env python3
"""
leida_occurrence_stats.py
=========================

Compute state-occurrence probabilities (per subject x condition x cluster)
for *every* K-means solution found in <results_root>/k_##/ and run pair-wise
statistics with a **global** multiplicity correction.

Outputs
-------
results_root/
└── stats_summary.csv                (#tests rows)
    k_##/
        occurrence.npy               (n_subj x n_cond x k)
        stats.json                   {"cond1__cond2": [[p_raw,p_adj,sig], …]}
        summary_barplot.png
        cluster_C00.png  …           (topomap + raincloud for every cluster)

Dependencies
------------
numpy, pandas, seaborn, matplotlib, statsmodels, scipy, (optional) mne
"""

from __future__ import annotations
import argparse, itertools, json, pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import fdrcorrection, multipletests

try:
    import mne
except ImportError:
    mne = None                   # topomaps will be skipped

# --------------------------------------------------------------------------- #
#                                command line                                 #
# --------------------------------------------------------------------------- #

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LEiDA state-occurrence statistics")
    p.add_argument("--results", required=True, type=Path,
                   help="Folder that contains k_## sub-folders")
    p.add_argument("--data", required=True, type=Path,
                   help="(unused at the moment - kept for compatibility)")
    p.add_argument("--conditions", nargs="+",
                   default=["Coordination", "Solo", "Spontaneous"],
                #    default=["SpontaneousSynchro", "SpontaneousNoSynchro"],
                   help="Condition names (must match keys in labels.pkl)")
    p.add_argument("--test", choices=["t", "logit", "wilcoxon"], default="logit",
                   help="'t' raw, 'logit' logit-transform + t-test, "
                        "'wilcoxon' signed-rank")
    p.add_argument("--corr", choices=["fdr", "bonferroni"], default="fdr",
                   help="Family-wise multiplicity correction")
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--epochs", type=Path, default="data/raw/PPT1/s_101_Coordination.set",
                   help="An EEGLAB/FIF/etc file to steal sensor positions from")
    p.add_argument("--montage", default="biosemi64",
                   help="Fallback montage if --epochs not given")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()

# --------------------------------------------------------------------------- #
#                            helper functions                                 #
# --------------------------------------------------------------------------- #

def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1. - p))

def paired_p(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Return p-value for paired samples `a`, `b` (mask *before* call!)."""
    if method == "wilcoxon":
        try:
            return wilcoxon(a, b, zero_method="wilcox").pvalue
        except ValueError:             # all zeros etc.
            return 1.0
    if method == "logit":
        a, b = logit(a), logit(b)
    return ttest_rel(a, b).pvalue

def collect_subjects(k_dirs: Sequence[Path],
                     conditions: Sequence[str]) -> list[str]:
    """Union of subject IDs appearing in any labels.pkl."""
    subjects: set[str] = set()
    for kd in k_dirs:
        with (kd / "labels.pkl").open("rb") as f:
            labels = pickle.load(f)
            print(f"Found {len(labels)} conditions in {kd}")
        for c in conditions:
            subjects.update(labels.get(c, {}).keys())
    return sorted(subjects)

# --------------------------------------------------------------------------- #
#                              plotting                                       #
# --------------------------------------------------------------------------- #

def topomap(ax: plt.Axes, center_vec: np.ndarray, info):
    if mne is None or info is None:
        ax.axis("off")
        ax.set_title("no sensors", fontsize=7)
        return
    pos_idx = np.where(center_vec > 0)[0]
    other   = [i for i in range(len(center_vec)) if i not in pos_idx]
    cmap    = matplotlib.colors.ListedColormap(["red", "lightgrey"])
    mne.viz.plot_sensors(info, kind="topomap",
                         ch_groups=[pos_idx, other],
                         cmap=cmap, axes=ax, show=False,
                         show_names=True, linewidth=.5)

def raincloud(ax: plt.Axes,
              occ: np.ndarray,       # subj × cond
              conds: list[str],
              colors: dict[str,str],
              pair_p: dict[tuple[str,str], tuple[float,bool]],
              title: str):
    """
    Draw violin + box + scatter of occ[:,ci] for ci in conds,
    and if a pair is sig, annotate BOTH star and p-value.
    Now also draws very faint lines connecting each subject’s points.
    """
    n_subj, n_conds = occ.shape
    X = np.arange(n_conds)

    # 1) pre-compute the exact x-offset for each subj×cond
    jitter = np.full((n_subj, n_conds), np.nan)
    for ci in range(n_conds):
        mask = ~np.isnan(occ[:, ci])
        jitter[mask, ci] = np.random.uniform(ci-0.35, ci-0.2, size=mask.sum())

    # 2) draw faint spaghetti lines using those x-offsets
    for si in range(n_subj):
        ys = occ[si, :]
        xs = jitter[si, :]
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        if valid.sum() > 1:
            ax.plot(xs[valid], ys[valid],
                    color='grey', alpha=0.2, linewidth=0.5, zorder=1)

    # 3) violins + boxes + scatter (using the same jitters)
    ymax = np.nanmax(occ)
    for i, cond in enumerate(conds):
        data = occ[:, i][~np.isnan(occ[:, i])]
        # violin
        v = ax.violinplot(data, [i], widths=0.7, showmeans=True,
                          showmedians=False, showextrema=False)
        body = v["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:,0] = np.clip(verts[:,0], i, i+0.4)
        body.set_facecolor(colors[cond]); body.set_alpha(.5); body.set_edgecolor("none")

        # boxplot
        ax.boxplot(data, positions=[i-0.15], widths=0.12, vert=True,
                   showcaps=False, patch_artist=True,
                   boxprops=dict(facecolor=colors[cond],alpha=.7),
                   medianprops=dict(color='k'))

        # scatter at the jittered x-positions
        subj_idx = np.where(~np.isnan(occ[:, i]))[0]
        xs = jitter[subj_idx, i]
        ys = occ[subj_idx, i]
        ax.scatter(xs, ys, s=15, c="k", alpha=.6, zorder=2)

    # 4) significance annotations (unchanged)
    y0, dy = ymax + 0.02, ymax * 0.06
    for idx, ((c1, c2), (p_adj, sig)) in enumerate(pair_p.items()):
        if not sig:
            continue
        x1, x2 = conds.index(c1), conds.index(c2)
        y = y0 + idx * dy
        ax.plot([x1, x1, x2, x2],
                [y, y + dy/3, y + dy/3, y],
                c="grey", lw=1.3)
        stars = "***" if p_adj < .001 else "**" if p_adj < .01 else "*"
        ax.text((x1 + x2) / 2, y + dy/3, stars,
                ha="center", va="bottom", color="r", fontsize=20)
        ax.text((x1 + x2) / 2, y + dy/3, f"p={p_adj:.3g}",
                ha="center", va="top", color="grey", fontsize=9)

    ax.set_xticks(X)
    ax.set_xticklabels(conds, fontsize=10)
    ax.set_ylabel("Occurrence", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", ls="--", alpha=.4)
    sns.despine(ax=ax, trim=True)

def barplot_all(folder: Path,
                occ: np.ndarray,             # subj × cond × k
                conds: Sequence[str],
                colors: dict[str,str],
                stats_json: dict[str, list[list]],
                dpi: int = 150):
    """
    Mean ± SEM per cluster, grouped by condition,
    with 1/2/3-star brackets between bars for each sig. cond. pair.
    """
    # --- compute means & sems --------------------------------------------
    k = occ.shape[-1]
    means = np.nanmean(occ, axis=0)      # cond × k
    sem   = np.nanstd(occ, axis=0) / np.sqrt(np.sum(~np.isnan(occ), axis=0))

    # bar geometry
    width = .8 / len(conds)
    x     = np.arange(k)

    # --- start figure ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.3*k), 4))

    # draw each condition’s bars
    for ci, cond in enumerate(conds):
        ax.bar(
            x + (ci - .5*(len(conds)-1))*width,
            means[ci], width,
            yerr=sem[ci], capsize=3,
            color=colors[cond], label=cond
        )

    # --- now add significance brackets ----------------------------------
    # find vertical spacing
    top_vals = (means + sem).max(axis=0)    # array of length k
    global_max = top_vals.max()
    dy = global_max * 0.05
    y_base = global_max + dy

    # stats_json keys are "cond1__cond2" → list of [p_raw,p_adj,sig] per cluster
    pairs = list(stats_json.keys())

    for c in range(k):
        # collect only the significant pairs for this cluster
        sigs = []
        for pair in pairs:
            p_raw, p_adj, sig = stats_json[pair][c]
            if sig:
                sigs.append((pair, p_adj))
        # draw them one above the other
        for idx, (pair, p_adj) in enumerate(sigs):
            c1, c2 = pair.split("__")
            i1, i2 = conds.index(c1), conds.index(c2)
            x1 = x[c] + (i1 - .5*(len(conds)-1))*width
            x2 = x[c] + (i2 - .5*(len(conds)-1))*width
            y  = y_base + idx*dy

            # little bracket
            ax.plot(
                [x1, x1, x2, x2],
                [y,   y+dy/3, y+dy/3, y],
                c="grey", lw=1.3
            )

            # star(s)
            stars = "***" if p_adj < .001 else "**" if p_adj < .01 else "*"
            ax.text(
                (x1 + x2)/2,
                y + dy/3,
                stars,
                ha="center", va="bottom",
                color="r", fontsize=14
            )

    # --- finish styling and save ----------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{ci}" for ci in x], fontsize=8)
    ax.set_ylabel("Mean occurrence ± SEM")
    ax.set_title("Occurrence probability – all clusters")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", ls="--", alpha=.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(folder / "summary_barplot.png", dpi=dpi)
    plt.close(fig)


import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import ticker
import mne

def plot_cluster_grid(results_root: Path,
                      conds: list[str],
                      cond_pair: tuple[str,str],
                      info,         # your mne.Info for sensor positions
                      dpi: int = 150):
    """
    Draw one big grid of topomaps for cond_pair across all k's.
    """
    # 1) find all k_dirs in ascending order
    k_dirs = sorted([d for d in results_root.glob("k_*") if d.is_dir()],
                    key=lambda p: int(p.name.split("_")[1]))
    ks = [int(d.name.split("_")[1]) for d in k_dirs]

    # 2) load everything and compute Δ's
    all_deltas = []
    data = {}
    c1, c2 = cond_pair
    key = f"{c1}__{c2}"
    idx1 = conds.index(c1)
    idx2 = conds.index(c2)

    for k, k_dir in zip(ks, k_dirs):
        occ     = np.load(k_dir / "occurrence.npy")     # subj × cond × cluster
        centers = np.load(k_dir / "centers.npy")        # cluster × sensors
        stats   = json.loads((k_dir / "stats.json").read_text())
        # compute mean difference per cluster
        deltas = [
            float(np.nanmean(occ[:, idx1, c]) - np.nanmean(occ[:, idx2, c]))
            for c in range(occ.shape[-1])
        ]
        all_deltas.extend(deltas)
        data[k] = dict(occ=occ, centers=centers, stats=stats[key], deltas=deltas)

    # 3) build a global colormap
    vmax = max(abs(d) for d in all_deltas)
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    # 4) prepare subplots
    max_k = max(ks)
    nrows = len(ks)
    fig, axes = plt.subplots(nrows, max_k,
                             figsize=(2*max_k, 2*nrows),
                             squeeze=False)

    for row, k in enumerate(ks):
        occ     = data[k]["occ"]
        centers = data[k]["centers"]
        stats   = data[k]["stats"]    # list of [p_raw,p_adj,sig] per cluster
        deltas  = data[k]["deltas"]
        nclust  = centers.shape[0]

        for c in range(max_k):
            ax = axes[row, c]
            # empty if this k has fewer clusters
            if c >= nclust:
                ax.axis("off")
                continue

            # pick accent color
            p_raw, p_adj, sig = stats[c]
            delta = deltas[c]
            accent = cmap(norm(delta)) if sig else "lightgrey"

            # split sensors by sign of center weight (just as before)
            pos_idx = np.where(centers[c] > 0)[0]
            other   = [i for i in range(centers.shape[1]) if i not in pos_idx]
            cmap2   = matplotlib.colors.ListedColormap([accent, "white"])

            # draw the topomap
            mne.viz.plot_sensors(info, kind="topomap",
                                 ch_groups=[pos_idx, other],
                                 cmap=cmap2, axes=ax,
                                 show=False, show_names=False,
                                 linewidth=0.5)

            # title: two means
            m1 = np.nanmean(occ[:, idx1, c])
            m2 = np.nanmean(occ[:, idx2, c])
            ax.text(0.2, 0.87, f"{m1:.2f}",
                    color="red",  # or cond_colors[c1]
                    ha="center", va="bottom",
                    transform=ax.transAxes,
                    fontsize=8)
            ax.text(0.8, 0.87, f"{m2:.2f}",
                    color="blue", # or cond_colors[c2]
                    ha="center", va="bottom",
                    transform=ax.transAxes,
                    fontsize=8)

            # underneath: stars
            if sig:
                stars = "***" if p_adj<.001 else "**" if p_adj<.01 else "*"
                ax.text(0.5, 1.05, stars,
                        color="green", fontsize=14,
                        ha="center", va="top",
                        transform=ax.transAxes)

            ax.set(xticks=[], yticks=[])

    # 5) a single colorbar on the right
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb  = matplotlib.colorbar.ColorbarBase(cax,
                cmap=cmap, norm=norm,
                orientation="vertical",
                ticks=ticker.MaxNLocator(5))
    cb.set_label(f"Δ({c1} – {c2})")

    fig.suptitle(f"{c1} vs {c2} – cluster-wise mean differences", y=0.98)
    fig.tight_layout(rect=[0,0,0.9,1])
    out = results_root / f"{c1}_vs_{c2}_cluster_grid.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Saved grid to {out}")



# --------------------------------------------------------------------------- #
#                               main                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    args          = _cli()
    k_dirs        = sorted((d for d in args.results.glob("k_*")
                            if d.is_dir()),
                           key=lambda p: int(p.name.split("_")[1]))
    ks            = [int(p.name.split("_")[1]) for p in k_dirs]
    subjects      = collect_subjects(k_dirs, args.conditions)
    n_subj        = len(subjects)
    cond_colors   = dict(zip(args.conditions,
                             sns.color_palette("Set2",
                                              len(args.conditions))))

    # montage for topomaps ---------------------------------------------------
    info = None
    if mne and args.epochs and args.epochs.exists():
        info = (mne.io.read_epochs_eeglab(args.epochs)
                      .set_montage(args.montage).info)

    # -----------------------------------------------------------------------
    #  pass 1 – build list of ALL tests  -> raw p list
    # -----------------------------------------------------------------------
    all_tests: list[tuple[int,int,str,str,float]] = []     # (k,c,cond1,cond2,p)
    per_k_raw: dict[int, dict[tuple[str,str], list[float]]] = {}

    for k, k_dir in zip(ks, k_dirs):
        # labels.pkl  --------------------------------------------------------
        with (k_dir / "labels.pkl").open("rb") as f:
            labels = pickle.load(f)

        # subj × cond × k   --------------------------------------------------
        occ = np.full((n_subj, len(args.conditions), k), np.nan)
        for ci, cond in enumerate(args.conditions):
            for si, subj in enumerate(subjects):
                if subj not in labels.get(cond, {}):
                    continue
                flat = labels[cond][subj].ravel()
                for c in range(k):
                    occ[si, ci, c] = (flat == c).mean()
        np.save(k_dir / "occurrence.npy", occ)

        # raw p-values -------------------------------------------------------
        per_k_raw[k] = {pair: [] for pair in itertools.combinations(args.conditions, 2)}
        for c in range(k):
            for pair in itertools.combinations(args.conditions, 2):
                a = occ[:, args.conditions.index(pair[0]), c]
                b = occ[:, args.conditions.index(pair[1]), c]
                mask = ~np.isnan(a) & ~np.isnan(b)
                p    = paired_p(a[mask], b[mask], args.test) if mask.sum() > 1 else 1.0
                per_k_raw[k][pair].append(p)
                all_tests.append((k, c, *pair, p))

    # -----------------------------------------------------------------------
    #  global multiplicity correction
    # -----------------------------------------------------------------------
    raw_p  = np.array([t[-1] for t in all_tests])
    if args.corr == "fdr":
        reject_all, p_adj_all = fdrcorrection(raw_p, alpha=args.alpha)
    else:                         # bonferroni
        reject_all, p_adj_all, _, _ = multipletests(raw_p,
                                                   alpha=args.alpha,
                                                   method="bonferroni")

    # map (k,c,cond1,cond2) -> (p_adj, sig)
    key_map = {(k,c,c1,c2): (p_adj, bool(sig))
               for (k,c,c1,c2,_), p_adj, sig
               in zip(all_tests, p_adj_all, reject_all)}

    # -----------------------------------------------------------------------
    #  pass 2 – write json / plots / summary rows
    # -----------------------------------------------------------------------
    summary_rows = []

    for k, k_dir in zip(ks, k_dirs):
        occ   = np.load(k_dir / "occurrence.npy")
        k_raw = per_k_raw[k]

        # ---------- stats.json ---------------------------------------------
        stats_json = {}
        sig_any_cluster = np.zeros(k, dtype=bool)

        for pair in k_raw:
            c1, c2     = pair
            rows_pair  = []
            for c, p_raw in enumerate(k_raw[pair]):
                p_adj, sig = key_map[(k, c, c1, c2)]
                rows_pair.append([float(p_raw), float(p_adj), bool(sig)])
                sig_any_cluster[c] |= sig        # any significant comparison?
                summary_rows.append(dict(k=k, cluster=c,
                                         cond1=c1, cond2=c2,
                                         p_raw=p_raw, p_corrected=p_adj,
                                         significant=sig))
            stats_json[f"{c1}__{c2}"] = rows_pair

        (k_dir / "stats.json").write_text(json.dumps(stats_json, indent=2))

        # ---------- bar-plot with significance summary ---------------------
        # barplot_all(k_dir, occ, args.conditions, cond_colors,
        #             sig_any_cluster.tolist(), dpi=args.dpi)

        barplot_all(k_dir, occ, args.conditions, cond_colors, stats_json, dpi=args.dpi)

        # ---------- cluster-wise combined plots ----------------------------
        centers = np.load(k_dir / "centers.npy")
        for c in range(k):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={"width_ratios":[1,2]})

            topomap(ax1, centers[c], info)

            pair_pvals = {(c1,c2): key_map[(k,c,c1,c2)] for c1,c2
                          in itertools.combinations(args.conditions, 2)}

            raincloud(ax2, occ[:, :, c], args.conditions,
                      cond_colors, pair_pvals, f"Cluster {c}")

            fig.tight_layout()
            fig.savefig(k_dir / f"cluster_C{c:02d}.png", dpi=args.dpi)
            plt.close(fig)

    # -----------------------------------------------------------------------
    pd.DataFrame(summary_rows).to_csv(args.results / "stats_summary.csv",
                                      index=False)
    # --------------------- generate cluster‐grid plots ---------------------
    # for every pair of conditions
    for cond1, cond2 in itertools.combinations(args.conditions, 2):
        plot_cluster_grid(
            results_root = args.results,
            conds        = args.conditions,
            cond_pair    = (cond1, cond2),
            info         = info,
            dpi          = args.dpi
        )
    # ------------------------------------------------------------------------
    sig_df = pd.DataFrame(summary_rows).query("significant")
    if len(sig_df):
        print("\nSignificant after correction:")
        print(sig_df.to_string(index=False))
    else:
        print("\nNo significant differences after correction.")

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()

# python src/mirror_leida/leida_occurence_stats.py --results data/kmeans/leading_eeg/alpha --data data/leading_eeg/alpha --corr bonferroni