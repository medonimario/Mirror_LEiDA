#!/usr/bin/env python3
"""
plot_leida_clusters.py
----------------------

Generate three diagnostic plots per K-means solution:
  1) 3-D PCA scatter coloured by cluster
  2) bar-plot of centroid weights
  3) (optional) EEG topomaps of positive-weight channels

Requires
--------
numpy, matplotlib, scikit-learn, joblib
MNE-Python (only if you ask for --epochs / --montage)
"""

from __future__ import annotations
import argparse, pickle, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  needed for 3-D proj

# optional MNE
try:
    import mne
except ImportError:
    mne = None


# ------------- helpers from the batch runner ---------------------------------
def load_leading_eigenvectors(data_dir, possible_conditions = ("Coordination", "Solo", "Spontaneous")):
    
    data_dict = {}
    for f in sorted(data_dir.glob("*.npy")):
        if "eigenvectors" not in f.name:
            continue
        base = f.stem[2:] if f.stem.startswith("s_") else f.stem
        subj, rest = base.split("_", 1)
        cond_raw = rest.split("-")[0]
        condition = next((c for c in possible_conditions
                          if cond_raw.lower().startswith(c.lower())), None)
        if condition is None:
            continue
        data_dict.setdefault(condition, {})[subj] = np.load(f)
    return data_dict


def collate_eigenvectors(data_dict):
    coll, meta = [], []
    for cond, subj_map in data_dict.items():
        for subj, eig in subj_map.items():
            n_epochs, n_windows, n_chan = eig.shape
            coll.append(eig.reshape(-1, n_chan))
            meta.extend((cond, subj, e, w)
                        for e in range(n_epochs) for w in range(n_windows))
    return np.vstack(coll).astype(np.float32), meta


# --------------------------- plotting ----------------------------------------
def plot_pca_3d(out_png, X, labels, centers, max_points=1000, title=""):
    N = len(X)
    if N > max_points:
        idx = []
        for k in np.unique(labels):
            where = np.where(labels == k)[0]
            take = max(1, int(max_points * len(where) / N))
            idx.extend(np.random.choice(where, take, replace=False))
        idx = np.array(idx)
        X, labels = X[idx], labels[idx]

    pca = PCA(n_components=3)
    Xp = pca.fit_transform(X)
    Cp = pca.transform(centers)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(*Xp.T, c=labels, cmap='rainbow', alpha=.55, s=20)
    ax.scatter(*Cp.T, c=np.arange(len(centers)), cmap='rainbow',
               marker='^', s=180, edgecolors='k', linewidths=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    var = pca.explained_variance_ratio_ * 100
    ax.text2D(.01, .99, f"Explained var: {var[0]:.1f}% / {var[1]:.1f}% / {var[2]:.1f}%",
              transform=ax.transAxes, va="top")
    fig.colorbar(sc, ax=ax, pad=.08, fraction=.04, label="cluster")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_centers_bar(out_png, centers, roi_names=None):
    k, n = centers.shape
    fig, axes = plt.subplots(k, 1, figsize=(min(10, n*.35), 1.5*k), sharex=False)
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        v = centers[i]
        colors = ['red' if x >= 0 else 'blue' for x in v]
        ax.bar(range(n), v, color=colors)
        ax.axhline(0, color='k', lw=.8)
        ax.set_ylabel(f"C{i}", rotation=0, labelpad=15, fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        if roi_names is not None:
            ax.set_xticks(range(n))
            ax.set_xticklabels(roi_names, rotation=90, fontsize=5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# def plot_topomaps(out_png, centers, epochs):
#     if mne is None:
#         print("⚠︎  MNE-Python not installed → skipping topomaps")
#         return
#     k, n = centers.shape
#     n_rows, n_cols = int(np.ceil(k / 5)), 5
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))
#     axes = axes.flatten()
#     for c in range(k):
#         vec = centers[c]


#         pos_idx = np.where(vec > 0)[0]
#         print(f"We have {len(pos_idx)} positive-weight channels in C{c} "
#               f"out of {n} total channels.")
#         other   = [i for i in range(n) if i not in pos_idx]
#         cmap = mcolors.ListedColormap(['red', 'lightgray'])
#         mne.viz.plot_sensors(
#             epochs.info, kind='topomap', show_names=False,
#             ch_groups=[pos_idx, other], cmap=cmap,
#             axes=axes[c], show=False, linewidth=.5)
#         axes[c].set_title(f"C{c}", fontsize=9)
#     for ax in axes[k:]:
#         ax.axis("off")
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=150)
#     plt.close(fig)

def plot_topomaps(out_png, centers, epochs):
    if mne is None:
        print("⚠︎  MNE-Python not installed → skipping topomaps")
        return

    k, n = centers.shape
    n_rows, n_cols = int(np.ceil(k / 5)), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
    axes = axes.flatten()

    for c in range(k):
        vec = centers[c]

        # Only keep positive weights; set negatives to zero
        pos_vals = np.copy(vec)
        pos_vals[pos_vals < 0] = 0.0

        # Count how many channels are truly > 0 (for info)
        n_pos = np.sum(vec > 0)
        print(f"We have {n_pos} positive‐weight channels in C{c} out of {n} total channels.")

        # Plot a continuous topomap of the positive‐only values
        #   - cmap="Reds" will go from white→red by intensity
        #   - vmin=0, vmax=max(pos_vals) ensures the full range is used
        mne.viz.plot_topomap(
            vec,
            epochs.info,
            axes=axes[c],
            show=False,
            sensors=True,
            # cmap="Reds",
            cmap='bwr',
            vlim=(np.min(vec),np.max(vec)),
            contours=0,
        )
        axes[c].set_title(f"C{c}", fontsize=9)

    # Turn off any unused subplots
    for ax in axes[k:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)



def align_centers_sign(centers):
    # Flip sign if number of positive values exceeds negative values
    new_centers = centers.copy()
    for i, c in enumerate(new_centers):
        if np.sum(c > 0) > np.sum(c < 0):
            new_centers[i] *= -1
    return new_centers



# ----------------------- CLI --------------------------------------------------
def cli():
    p = argparse.ArgumentParser(description="Plot LEiDA K-means results")
    p.add_argument("--input",  required=True, type=Path,
                   help="directory with *_eigenvectors.npy (to rebuild PCA)")
    p.add_argument("--results", required=True, type=Path,
                   help="directory containing k_## folders from the batch run")
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="specific K values (default: all k_## dirs present)")
    p.add_argument("--epochs", type=Path, default="data/raw/PPT1/s_101_Coordination.set",
                   help="an EEGLAB .set, FIF, or NPZ file with channel locations "
                        "(enables topomaps).")
    p.add_argument("--montage", type=str, default='biosemi64',
                   help='if you have no epochs file, give a montage name '
                        '(e.g. "biosemi64"); only positive-weight channels will '
                        'be plotted on the template head.')
    p.add_argument("--conditions", type=str, nargs="+",
                   default=["Coordination", "Solo", "Spontaneous"],
                   help="conditions to load from the input directory")
    return p.parse_args()


def main():
    args = cli()

    # PCA needs collated eigenvectors
    data_dict = load_leading_eigenvectors(args.input, args.conditions)
    coll_eigs, _ = collate_eigenvectors(data_dict)

    # channel names for bar-plots
    roi_names = None
    if args.epochs:
        ep = mne.io.read_epochs_eeglab(args.epochs)
        ep.set_montage(args.montage)
        roi_names = ep.ch_names
    elif args.montage and mne is not None:
        ep.set_montage(args.montage)
        info = mne.create_info(ch_names=[f"R{i}" for i in range(coll_eigs.shape[1])],
                               sfreq=1.0, ch_types="eeg")
        info.set_montage(args.montage)
        ep = mne.EpochsArray(np.zeros((1, len(info.ch_names), 1)), info)
    else:
        ep = None

    # which Ks?
    if args.k is None:
        args.k = sorted(int(p.name.split("_")[1]) for p in args.results.glob("k_*"))

    for k in args.k:
        folder = args.results / f"k_{k:02d}"
        if not folder.exists():
            print(f"⚠︎  {folder} not found, skipping")
            continue

        centers = np.load(folder / "centers.npy")
        centers = align_centers_sign(centers) # ensure consistent sign



        with open(folder / "labels.pkl", "rb") as f:
            labels_dict = pickle.load(f)
        # flatten labels_dict to match coll_eigs order
        labels = np.concatenate([v.reshape(-1)
                                 for cond in labels_dict.values()
                                 for v in cond.values()])

        # ---------- plots ----------
        plot_pca_3d(folder / "pca_clusters.png",
                    coll_eigs, labels, centers,
                    title=f"K={k} – PCA of Eigen-vectors")

        plot_centers_bar(folder / "centers_barplot.png",
                         centers, roi_names)

        if ep is not None:
            plot_topomaps(folder / "centers_topomaps.png",
                          centers, ep)

        print(f"✓  plots written to {folder}")

    print("done.")


if __name__ == "__main__":
    main()

#python src/mirror_leida/plot_leida_clusters.py --input data/leading_eeg_spontaneous/alpha --results data/kmeans/leading_eeg_spontaneous/alpha