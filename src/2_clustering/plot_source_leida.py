#!/usr/bin/env python3
"""
plot_source_leida.py
--------------------

Generate diagnostic plots for LEiDA clustering results on source-level (ROI) data.
This version uses a custom, high-quality brain visualization function.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mne

# ---------------------------------------------------------------------
# 1. Data Loading and Collation Helpers (Unchanged)
# ---------------------------------------------------------------------
def load_and_collate_eigenvectors(file_paths: list[Path]) -> np.ndarray:
    all_eigs_list = [np.load(f).reshape(-1, np.load(f).shape[-1]) for f in file_paths]
    return np.vstack(all_eigs_list)

def flatten_labels_by_file_order(labels_dict: dict, file_paths: list[Path], possible_conditions: list[str]) -> np.ndarray:
    flat_labels = []
    for f_path in file_paths:
        parts = f_path.stem.split('_')
        if len(parts) < 3: continue
        subj, cond_raw = parts[1], parts[2].split('-')[0]
        condition = next((c for c in possible_conditions if cond_raw.lower() == c.lower()), None)
        if condition is None: raise KeyError(f"Label data not found for file {f_path.name}")
        flat_labels.append(labels_dict[condition][subj].flatten())
    return np.concatenate(flat_labels)

def get_roi_names_from_source_file(source_fif_dir: Path) -> list[str] | None:
    first_file = next(source_fif_dir.glob("*-source-beamformer-epo.fif"), None)
    if not first_file: return None
    return mne.read_epochs(first_file, verbose='ERROR').ch_names

# ---------------------------------------------------------------------
# 2. Plotting Functions
# ---------------------------------------------------------------------
def align_centers_to_minority_positive(centers: np.ndarray) -> np.ndarray:
    aligned_centers = centers.copy()
    for i, center_vec in enumerate(aligned_centers):
        if np.sum(center_vec > 0) > np.sum(center_vec < 0):
            aligned_centers[i] *= -1
    return aligned_centers

def plot_pca_3d(out_png: Path, X: np.ndarray, labels: np.ndarray, centers: np.ndarray, max_points=2000, title=""):
    # (Your preferred PCA plot function, unchanged)
    n_samples = X.shape[0]
    if n_samples > max_points:
        sampled_indices = []
        unique_labels, counts = np.unique(labels, return_counts=True)
        for k, count in zip(unique_labels, counts):
            cluster_indices = np.where(labels == k)[0]
            sample_size = max(1, int(round(count * max_points / n_samples)))
            if sample_size < count:
                sampled_cluster_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
            else:
                sampled_cluster_indices = cluster_indices
            sampled_indices.extend(sampled_cluster_indices)
        sampled_indices = np.array(sampled_indices)
        plot_vectors, plot_labels = X[sampled_indices], labels[sampled_indices]
        subsample_info = f"Showing {len(plot_vectors)} of {n_samples} points"
    else:
        plot_vectors, plot_labels = X, labels
        subsample_info = f"Showing all {n_samples} points"
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(plot_vectors)
    centers_pca = pca.transform(centers)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=plot_labels, cmap='rainbow', alpha=0.6, s=30, rasterized=True)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], c=np.arange(len(centers)), cmap='rainbow', marker='^', s=250, edgecolors='k', linewidths=1.5, label='Cluster Centroids')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.grid(True)
    explained_var = pca.explained_variance_ratio_ * 100
    ax.text2D(0.02, 0.98, f"Explained Var: {explained_var[0]:.1f}%, {explained_var[1]:.1f}%, {explained_var[2]:.1f}%", transform=ax.transAxes, va='top', ha='left', fontsize=9)
    ax.text2D(0.02, 0.93, subsample_info, transform=ax.transAxes, va='top', ha='left', fontsize=9)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label("Cluster ID")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_centers_bar(out_png: Path, centers: np.ndarray, roi_names: list[str]):
    # (This function is correct, unchanged)
    k, n_rois = centers.shape
    fig, axes = plt.subplots(k, 1, figsize=(min(12, n_rois * 0.18), 1.5 * k), sharex=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        v = centers[i]
        colors = ['#d62728' if x >= 0 else '#1f77b4' for x in v]
        ax.bar(range(n_rois), v, color=colors, width=0.8)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel(f"C{i}", rotation=0, labelpad=20, fontsize=9, ha='right', va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.set_xlim(-0.5, n_rois - 0.5)
    axes[-1].set_xticks(range(n_rois))
    axes[-1].set_xticklabels(roi_names, rotation=90, fontsize=6)
    fig.suptitle("Cluster Centroid ROI Weights", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# =========================================================================
# YOUR CUSTOM BRAIN VISUALIZATION FUNCTION (ADAPTED)
# =========================================================================
def plot_brain_clusters_custom(out_png: Path, centers: np.ndarray, all_mne_labels: list[mne.Label], ordered_roi_names: list[str], subjects_dir: Path):
    """
    Visualizes brain clusters by highlighting the minority positive ROIs, based on your custom design.
    """
    n_clusters = centers.shape[0]
    view_angles = ['lateral-left', 'lateral-right', 'dorsal', 'ventral']
    
    # Your hand-picked vibrant colors
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33',
              '#a65628','#f781bf','#1b9e77','#d95f02','#7570b3','#e6ab02']
    
    fig, axes = plt.subplots(n_clusters, len(view_angles), figsize=(2.5 * len(view_angles), 2.2 * n_clusters))
    # Handle case of K=1
    if n_clusters == 1:
        axes = np.array([axes])

    for c in range(n_clusters):
        center_vec = centers[c, :]
        pos_indices = np.where(center_vec > 0)[0]
        pos_roi_names = {ordered_roi_names[i] for i in pos_indices}
        
        # This mapping is crucial: MNE labels are not ordered, so we must find them by name
        labels_to_plot = [lbl for lbl in all_mne_labels if lbl.name in pos_roi_names]
        print(f"Cluster {c}: Plotting {len(labels_to_plot)} positive-weight ROIs.")
        
        # Use a single Brain object and switch views for efficiency
        brain = mne.viz.Brain("fsaverage", hemi="both", surf="pial", subjects_dir=subjects_dir,
                              background="white", size=(400, 400), cortex='low_contrast')
        
        # Add all positive labels to the brain object once
        color = colors[c % len(colors)]
        for label in labels_to_plot:
            brain.add_label(label, color=color, alpha=0.9, borders=False)
            brain.add_label(label, color='black', alpha=0.7, borders=True)
            
        for j, view in enumerate(view_angles):
            if view == 'lateral-left': brain.show_view('lateral', hemi='lh')
            elif view == 'lateral-right': brain.show_view('lateral', hemi='rh')
            else: brain.show_view(view)
            
            img = brain.screenshot()
            ax = axes[c, j]
            ax.imshow(img)
            ax.set_axis_off()

            if c == 0:
                ax.set_title(view.replace('-', ' ').capitalize(), fontsize=12)
            
            if j == 0:
                ax.text(-0.1, 0.5, f"Cluster {c}", fontsize=12, ha='right', va='center',
                        transform=ax.transAxes,
                        bbox=dict(facecolor=color, alpha=0.5, boxstyle='round,pad=0.4'))
        brain.close()

    fig.suptitle(f"LEiDA States (K={n_clusters})", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.92) # Adjust layout for labels
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------
# 3. Main Driver (Calls the new plotting function)
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot LEiDA clustering results from source-level data.")
    # ... (parser arguments are correct, unchanged) ...
    parser.add_argument("--input_eigs", required=True, type=Path, help="Directory with original *_eigenvectors.npy files")
    parser.add_argument("--results", required=True, type=Path, help="Directory containing k_## folders from the clustering run")
    parser.add_argument("--source_fif_dir", required=True, type=Path, help="Directory with source-level .fif files")
    parser.add_argument("--subjects_dir", type=Path, required=True, help="Path to FreeSurfer subjects directory")
    parser.add_argument("--parc", type=str, default="aparc", help="Parcellation used (e.g., 'aparc')")
    parser.add_argument("--k", type=int, nargs="+", help="Specific K values to plot (default: all found)")
    args = parser.parse_args()

    # --- Setup ---
    eigs_dir, results_dir = args.input_eigs, args.results
    possible_conditions = ["Coordination", "Solo", "Spontaneous"]
    sorted_file_paths = sorted(eigs_dir.glob("s_*_*-eigenvectors.npy"))
    if not sorted_file_paths:
        print(f"Error: No eigenvector files found in {eigs_dir}. Exiting.")
        return

    coll_eigs = load_and_collate_eigenvectors(sorted_file_paths)
    ordered_roi_names = get_roi_names_from_source_file(args.source_fif_dir)
    if not ordered_roi_names: return

    all_mne_labels = mne.read_labels_from_annot("fsaverage", parc=args.parc, subjects_dir=args.subjects_dir)
    # Remove the 'unknown' label if it exists
    all_mne_labels = [lbl for lbl in all_mne_labels if 'unknown' not in lbl.name]

    k_values = args.k if args.k else sorted([int(p.name.split('_')[1]) for p in results_dir.glob("k_*")])
    if not k_values:
        print(f"Error: No 'k_##' subdirectories found in {results_dir}")
        return

    # --- Loop over each K ---
    for k in k_values:
        k_folder = results_dir / f"k_{k:02d}"
        if not k_folder.exists():
            print(f"⚠️ Skipping K={k}, directory not found: {k_folder}")
            continue

        print(f"\n--- Processing K={k} ---")
        
        centers = np.load(k_folder / "centers.npy")
        with open(k_folder / "labels.pkl", "rb") as f:
            labels_dict = pickle.load(f)
        
        labels_1d = flatten_labels_by_file_order(labels_dict, sorted_file_paths, possible_conditions)
        if len(labels_1d) != len(coll_eigs):
            print(f"FATAL ERROR for K={k}: Mismatch between eigenvectors ({len(coll_eigs)}) and labels ({len(labels_1d)}).")
            continue

        aligned_centers = align_centers_to_minority_positive(centers)
        
        # --- Generate Plots ---
        print("  -> Plotting PCA...")
        plot_pca_3d(k_folder / "pca_clusters.png", coll_eigs, labels_1d, aligned_centers, title=f"K={k} – PCA of Eigenvectors")

        print("  -> Plotting center barplots...")
        plot_centers_bar(k_folder / "centers_barplot.png", aligned_centers, ordered_roi_names)

        print("  -> Plotting brain clusters...")
        # Call your custom function here
        plot_brain_clusters_custom(k_folder / "centers_brain_plots.png", aligned_centers, all_mne_labels, ordered_roi_names, args.subjects_dir)

        print(f"✓ Plots for K={k} saved to {k_folder}")

    print("\nAll plotting complete.")


if __name__ == "__main__":
    main()