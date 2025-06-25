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
    from matplotlib.colors import LinearSegmentedColormap
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

# --------------------------- plotting (with DEPRECATION FIX) ----------------------------------------
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

    # --- Discrete Color-mapping Setup ---
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    # Get a discrete colormap (using the modern, non-deprecated method)
    cmap = plt.get_cmap('rainbow', n_clusters) # <<< THIS LINE IS UPDATED
    # Define boundaries for the colors. Ticks will be centered between them.
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)


    # 3D PCA
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(X)
    Cp = pca.transform(centers)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Use the new cmap and norm for discrete coloring
    sc = ax.scatter(*Xp.T, c=labels, cmap=cmap, norm=norm, alpha=.55, s=20)
    ax.scatter(*Cp.T, c=np.arange(len(centers)), cmap=cmap, norm=norm,
               marker='^', s=180, edgecolors='k', linewidths=1.2)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    var = pca.explained_variance_ratio_ * 100
    ax.text2D(.01, .99, f"Explained var: {var[0]:.1f}% / {var[1]:.1f}% / {var[2]:.1f}%",
              transform=ax.transAxes, va="top")

    # --- Create a discrete colorbar ---
    # The ticks are set to the integer cluster IDs
    cbar = fig.colorbar(sc, ax=ax, pad=.08, fraction=.04, ticks=unique_labels)
    cbar.set_label("Cluster ID")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_pca_2d(out_png, X, labels, centers, max_points=1000, title=""):
    N = len(X)
    if N > max_points:
        idx = []
        for k in np.unique(labels):
            where = np.where(labels == k)[0]
            take = max(1, int(max_points * len(where) / N))
            idx.extend(np.random.choice(where, take, replace=False))
        idx = np.array(idx)
        X, labels = X[idx], labels[idx]
        
    # --- Discrete Color-mapping Setup ---
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    # Get a discrete colormap (using the modern, non-deprecated method)
    cmap = plt.get_cmap('rainbow', n_clusters) # <<< THIS LINE IS UPDATED
    # Define boundaries for the colors. Ticks will be centered between them.
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)


    # 2D PCA
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    Cp = pca.transform(centers)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use the new cmap and norm for discrete coloring
    sc = ax.scatter(Xp[:, 0], Xp[:, 1],
                    c=labels, cmap=cmap, norm=norm, alpha=0.6, s=25,
                    edgecolors='k', linewidths=0.2)
    ax.scatter(Cp[:, 0], Cp[:, 1],
               c=np.arange(len(centers)), cmap=cmap, norm=norm,
               marker='^', s=200, edgecolors='black', linewidths=1.2)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Explained variance annotation
    var = pca.explained_variance_ratio_ * 100
    ax.text(0.01, 0.99,
            f"Explained var: PC1 {var[0]:.1f}%  |  PC2 {var[1]:.1f}%",
            transform=ax.transAxes, va='top', fontsize=10)

    # --- Create a discrete colorbar ---
    # The ticks are set to the integer cluster IDs
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08, ticks=unique_labels)
    cbar.set_label("Cluster ID")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_centers_bar(out_png, centers, roi_names=None):
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    k, n = centers.shape
    fig, axes = plt.subplots(1, k, figsize=(3 * k, max(4, 0.25 * n)), sharey=True)
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        v = centers[i]
        colors = ['red' if x >= 0 else 'blue' for x in v]
        ax.barh(range(n), v, color=colors, height=0.8)
        ax.axvline(0, color='k', lw=.8)
        ax.set_title(f"Center {i}", fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.invert_yaxis()
    if roi_names is not None and k > 0:
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(roi_names, fontsize=8)
    else:
        axes[0].set_yticks([])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Cluster Center Profiles", fontsize=14)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_topomaps(out_png, centers, epochs):
    if mne is None:
        print("⚠︎  MNE-Python not installed → skipping topomaps")
        return

    k, n = centers.shape
    n_rows, n_cols = int(np.ceil(k / 5)), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    vmax = np.max(np.abs(centers))
    vmin = -vmax
    for c in range(k):
        mne.viz.plot_topomap(
            centers[c], epochs.info, axes=axes[c], show=False, sensors=True,
            names=epochs.ch_names, cmap='bwr', vlim=(vmin, vmax), contours=0,
        )
        axes[c].set_title(f"C{c}", fontsize=9)
    for ax in axes[k:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)




# ==================== NEW and IMPROVED PLOTTING FUNCTION =========================
def plot_brain_clusters(out_png, centers, mne_labels, k_val,
                        view_angles=('lateral', 'medial', 'dorsal', 'ventral')):
    """
    Visualize brain clusters by highlighting positive-valued ROIs on a 3D brain.

    The subplot orientation is (views x clusters). The color intensity of each
    highlighted ROI is proportional to its value in the cluster center vector.
    The cluster title includes a colored background for easy identification.

    Note: Assumes `centers` has been processed by `align_centers_sign`, so
    positive values represent the smaller group of opposite-phase ROIs.

    Parameters
    ----------
    out_png : Path
        Path to save the output PNG file.
    centers : np.ndarray
        Cluster centers from K-means, shape (n_clusters, n_rois).
    mne_labels : list[mne.Label]
        List of MNE Label objects, ordered to match the columns of `centers`.
    k_val : int
        The K-value for the current solution, used for the title.
    view_angles : tuple, optional
        Views for brain visualization, by default ('lateral', 'medial', 'dorsal', 'ventral').
    """
    if mne is None:
        print("⚠︎  MNE-Python not installed → skipping brain plots")
        return

    n_clusters, n_rois = centers.shape
    n_views = len(view_angles)

    vmax = np.max(centers)
    if vmax <= 0:
        print("⚠︎  No positive values found in any cluster center. Cannot generate brain plots.")
        return

    base_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                   '#a65628', '#f781bf', '#1b9e77', '#d95f02', '#7570b3', '#e6ab02']

    fig, axes = plt.subplots(n_views, n_clusters,
                           figsize=(3 * n_clusters, 2.5 * n_views),
                           gridspec_kw={'wspace': 0.05, 'hspace': 0.15}) # Increased hspace for title box
    if n_views == 1 and n_clusters == 1:
        axes = np.array([[axes]])
    elif n_views == 1:
        axes = axes[np.newaxis, :]
    elif n_clusters == 1:
        axes = axes[:, np.newaxis]

    Brain = mne.viz.get_brain_class()

    for c in range(n_clusters):
        center_vec = centers[c, :]
        pos_indices = np.where(center_vec > 0)[0]
        
        cluster_color_hex = base_colors[c % len(base_colors)]
        cluster_cmap = LinearSegmentedColormap.from_list(
            'custom_cluster_cmap', ['#FFFFFF', cluster_color_hex]
        )

        brain = Brain("fsaverage", hemi="both", surf="pial", background="white",
                      size=(400, 400), alpha=0.7)

        for idx in pos_indices:
            if idx < len(mne_labels):
                label_value = center_vec[idx]
                rgba_color = cluster_cmap(label_value / vmax)
                
                lab = mne_labels[idx]
                brain.add_label(lab, color=rgba_color, alpha=1.0, borders=False)
                brain.add_label(lab, color='black', alpha=0.8, borders=True)

        for j, view in enumerate(view_angles):
            brain.show_view(view)
            img = brain.screenshot()
            
            ax = axes[j, c]
            ax.imshow(img)
            ax.axis('off')

            # <<< CHANGE IS HERE: RESTORED THE COLORED BOX FOR THE TITLE >>>
            if j == 0:
                ax.set_title(
                    f"Cluster {c}",
                    fontsize=12,
                    pad=18,  # Add padding to lift the box off the image
                    bbox=dict(
                        facecolor=cluster_color_hex,
                        alpha=0.6,
                        edgecolor='none',
                        boxstyle='round,pad=0.4'
                    )
                )

            if c == 0:
                ax.text(-0.05, 0.5, view.capitalize(), fontsize=12,
                        ha='right', va='center', transform=ax.transAxes, rotation=90)
        
        brain.close()

    fig.suptitle(f"LEiDA Brain States (K={k_val})", fontsize=16)
    fig.tight_layout(rect=[0.03, 0, 1, 0.93]) # Adjust rect for suptitle and row labels
    
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def align_centers_sign(centers):
    # Flip sign so the majority of values in each center are negative.
    # This means the smaller set of "opposite-phase" ROIs will be positive.
    new_centers = centers.copy()
    for i, c in enumerate(new_centers):
        if np.sum(c > 0) > np.sum(c < 0):
            new_centers[i] *= -1
    return new_centers



# ----------------------- CLI --------------------------------------------------
def cli():
    p = argparse.ArgumentParser(description="Plot LEiDA K-means results")
    p.add_argument("--eigenvectors_dir",  required=True, type=Path,
                   help="directory with *-eigenvectors.npy (to rebuild PCA)")
    p.add_argument("--clusters_dir", required=True, type=Path,
                   help="directory containing k_## folders from the batch run")
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="specific K values (default: all k_## dirs present)")
    p.add_argument("--epochs", type=Path, default=None,
                   help="an EEGLAB .set, FIF, or other MNE-readable file with channel locations. "
                        "Required for topomaps and provides ROI names for other plots.")
    p.add_argument("--montage", type=str, default='biosemi64',
                   help='If --epochs is not given, specify a standard montage name.')
    p.add_argument("--conditions", type=str, nargs="+",
                   default=["Coordination", "Solo", "Spontaneous"],
                   help="conditions to load from the input directory")
    
    # --- NEW ARGUMENTS FOR BRAIN PLOTTING ---
    p.add_argument("--brain-plots", action="store_true",
                   help="Generate 3D brain visualizations showing positive-valued ROIs for each "
                        "cluster. Requires a FreeSurfer installation and --epochs.")
    p.add_argument("--subjects-dir", type=Path, default=Path.home() / "mne_data/MNE-fsaverage-data",
                   help="Path to FreeSurfer subjects directory (for brain plots).")
    p.add_argument("--parcellation", type=str, default="aparc",
                   help="Parcellation to use for brain plots (e.g., 'aparc', 'aparc.a2009s').")
    
    return p.parse_args()


def main():
    args = cli()

    # PCA needs collated eigenvectors
    print("Loading and collating eigenvectors...")
    data_dict = load_leading_eigenvectors(args.eigenvectors_dir, args.conditions)
    if not data_dict:
        print("✕ Error: No eigenvector files found or matched in the input directory. Exiting.")
        return
        
    coll_eigs, _ = collate_eigenvectors(data_dict)
    print(f"Loaded {coll_eigs.shape[0]} total samples from {coll_eigs.shape[1]} ROIs.")

    # --- Setup MNE objects (Epochs, ROI names, and Labels) ---
    ep, roi_names, matched_mne_labels, is_sensor_space = None, None, None, False
    if args.epochs and args.epochs.exists():
        if mne is None:
            print("⚠︎  MNE not found, cannot load --epochs file.")
        else:
            print(f"Loading info from {args.epochs}...")
            if args.epochs.suffix == ".fif":
                ep = mne.read_epochs(args.epochs, preload=False, verbose=False)
            elif args.epochs.suffix == ".set":
                ep = mne.io.read_epochs_eeglab(args.epochs, verbose=False)
                ep.set_montage(args.montage)
            else:
                raise ValueError(f"Unsupported epochs file format: {args.epochs.suffix}")

            # <<< CHANGE: Detect if data is from sensor space (EEG/MEG) or source space >>>
            if len(ep.ch_names) == 64:
                is_sensor_space = True
                print("Detected sensor-space data. Topomaps will be generated.")
                if ep.get_montage() is None:
                    print(f"No montage found, setting standard '{args.montage}' montage.")
                    ep.set_montage(args.montage, on_missing='warn')
            else:
                is_sensor_space = False
                print("Detected source-space data. Topomaps will be skipped.")
            
            roi_names = ep.ch_names
    
    # --- Setup for Brain Plots (requires MNE and fsaverage) ---
    if args.brain_plots:
        if mne is None:
            print("⚠︎  MNE-Python not installed → skipping brain plots.")
            args.brain_plots = False
        elif roi_names is None:
            print("⚠︎  --brain-plots requires --epochs to provide ROI names for matching. Skipping.")
            args.brain_plots = False
        elif not args.subjects_dir.exists():
            print(f"⚠︎  Subjects directory not found at '{args.subjects_dir}'. Skipping brain plots.")
            args.brain_plots = False
        else:
            print(f"Loading '{args.parcellation}' labels from fsaverage for brain plots...")
            mne.utils.set_config("SUBJECTS_DIR", str(args.subjects_dir), set_env=True)
            
            fsaverage_labels = mne.read_labels_from_annot("fsaverage", parc=args.parcellation)
            label_map = {label.name: label for label in fsaverage_labels}
            
            matched_mne_labels = []
            unmatched_rois = []
            for name in roi_names:
                # The user's ROI names ('bankssts-lh') already match the MNE label format
                if name in label_map:
                    matched_mne_labels.append(label_map[name])
                else:
                    unmatched_rois.append(name)
            
            if unmatched_rois:
                print(f"⚠︎  Warning: Could not find MNE labels for {len(unmatched_rois)}/{len(roi_names)} ROIs.")
                if len(unmatched_rois) < 10:
                    print(f"   Unmatched ROIs: {unmatched_rois}")
                else:
                    print(f"   Unmatched (first 5): {unmatched_rois[:5]}")

            if not matched_mne_labels:
                print("✕  Error: No ROIs could be matched to the parcellation. Disabling brain plots.")
                args.brain_plots = False
            else:
                print(f"✓  Successfully matched {len(matched_mne_labels)} ROIs to brain parcellation.")

    # which Ks?
    if args.k is None:
        k_dirs = sorted(args.clusters_dir.glob("k_*"))
        if not k_dirs:
            print(f"✕ Error: No 'k_*' directories found in {args.clusters_dir}. Exiting.")
            return
        args.k = [int(p.name.split("_")[1]) for p in k_dirs]

    for k in args.k:
        folder = args.clusters_dir / f"k_{k:02d}"
        if not folder.exists():
            print(f"⚠︎  {folder} not found, skipping")
            continue

        print(f"\n--- Processing K={k} ---")
        centers = np.load(folder / "centers.npy")
        centers = align_centers_sign(centers)

        with open(folder / "labels.pkl", "rb") as f:
            labels_dict = pickle.load(f)
        labels = np.concatenate([v.reshape(-1)
                                 for cond in labels_dict.values()
                                 for v in cond.values()])

        # ---------- plots ----------
        plot_pca_3d(folder / "clusters_pca3D.png",
                    coll_eigs, labels, centers,
                    title=f"K={k} - PCA of Eigenvectors")
        
        plot_pca_2d(folder / "clusters_pca2D.png",
                    coll_eigs, labels, centers,
                    title=f"K={k} - PCA of Eigenvectors (2D)")

        plot_centers_bar(folder / "clusters_barplot.png",
                         centers, roi_names)

        # <<< CHANGE: Only call plot_topomaps if data is sensor-level >>>
        if ep is not None and is_sensor_space:
            plot_topomaps(folder / "clusters_topomaps.png", centers, ep)
        
        if args.brain_plots and matched_mne_labels:
            plot_brain_clusters(folder / "clusters_brain.png", centers, matched_mne_labels, k_val=k)

        print(f"✓  Plots written to {folder}")

    print("\nDone.")

if __name__ == "__main__":
    main()

# python src/2_clustering/plot_clusters.py --eigenvectors_dir data/leading_source/all/beamformer/alpha_256 --clusters_dir data/clustering_source/diametrical/all/beamformer/alpha_256 --epochs data/archive/source/beamformer/s_101_Coordination-source-beamformer-epo.fif --brain-plots
# python src/2_clustering/plot_clusters.py --eigenvectors_dir data/leading_eeg/all/alpha_256 --clusters_dir data/clustering_eeg/kmeans/all/alpha_256 --epochs data/raw_eeg/raw_all/PPT1/s_101_Coordination.set