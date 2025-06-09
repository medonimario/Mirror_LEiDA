#!/usr/bin/env python3
"""
clustering.py
-----------------------

Unified script to run LEiDA-style clustering on leading eigenvectors.
Supports both standard K-Means (with sign-flipping) and Diametrical
clustering over a range of K values.

The script automatically constructs input and output paths based on
parameters matching the eigenvector generation script.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed, cpu_count

# Numba is essential for fast diametrical clustering
try:
    import numba
except ImportError:
    print("Warning: Numba not found. Diametrical clustering will be very slow.")
    # Create a dummy decorator if numba is not installed
    def numba_jit_dummy(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    numba.jit = numba_jit_dummy

# --- HELPER & PRE-PROCESSING FUNCTIONS ---

def flip_eigenvectors(coll_eigs: np.ndarray) -> np.ndarray:
    """
    Applies sign-flipping to each eigenvector for K-Means consistency.
    Flips the vector if the majority of its components are positive.
    (As described in Olsen et al., 2022, for traditional LEiDA).
    """
    print("Applying sign-flipping convention for K-Means...")
    flipped_eigs = coll_eigs.copy()
    flip_count = 0
    for i in range(flipped_eigs.shape[0]):
        V1 = flipped_eigs[i, :]
        if np.mean(V1 > 0) > 0.5:
            flipped_eigs[i, :] = -V1
            flip_count += 1
        elif np.mean(V1 > 0) == 0.5:
            if np.sum(V1[V1 > 0]) > -np.sum(V1[V1 < 0]):
                flipped_eigs[i, :] = -V1
                flip_count += 1
    print(f"Flipping complete. {flip_count:,} of {flipped_eigs.shape[0]:,} eigenvectors were flipped.")
    return flipped_eigs

def map_labels_back(labels_1d, meta, data_dict):
    """Reorganize 1D cluster labels back into a nested dictionary."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, (cond, subj, e, w) in zip(labels_1d, meta):
        if subj not in out.setdefault(cond, {}):
             out[cond][subj] = np.empty(data_dict[cond][subj].shape[:2], dtype=np.int16)
        out[cond][subj][e, w] = lab
    return out

# --- CLUSTERING ALGORITHMS ---

def run_kmeans_clustering(data: np.ndarray, K: int, n_init=50, max_iter=300, random_state=None):
    """Standard K-means with clusters re-labelled by descending size."""
    km = KMeans(
        n_clusters=K, n_init=n_init, max_iter=max_iter,
        random_state=random_state, verbose=0
    ).fit(data)
    counts = np.bincount(km.labels_, minlength=K)
    order = np.argsort(counts)[::-1]  # Biggest cluster first
    relabel_map = np.empty_like(order); relabel_map[order] = np.arange(K)
    
    return {
        "labels": relabel_map[km.labels_],
        "centers": km.cluster_centers_[order],
        "counts": counts[order],
        "distances": km.transform(data)[:, order].astype(np.float32)
    }

@numba.jit(nopython=True, fastmath=True, cache=True)
def _diametrical_single_run(X: np.ndarray, K: int, max_iter: int, seed: int):
    """A single, fast run of diametrical clustering, JIT-compiled by Numba."""
    n, p = X.shape
    np.random.seed(seed)
    mu = np.empty((K, p), dtype=X.dtype)
    mu[0, :] = X[np.random.randint(n), :]
    min_sq_dists = np.full(n, np.inf, dtype=X.dtype)
    for i in range(1, K):
        dist_sq = 1.0 - (X @ mu[i-1, :].T)**2
        min_sq_dists = np.minimum(min_sq_dists, dist_sq)
        s = min_sq_dists.sum()
        probs = min_sq_dists / s if s > 0 else np.full(n, 1.0/n, dtype=X.dtype)
        next_idx = np.searchsorted(np.cumsum(probs), np.random.rand())
        mu[i, :] = X[next_idx, :]

    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(max_iter):
        sims_sq = (X @ mu.T)**2
        new_labels = np.argmax(sims_sq, axis=1).astype(np.int32)
        if np.all(new_labels == labels): break
        labels = new_labels
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            if len(idx_k) == 0: continue
            A_k_mu_k = X[idx_k, :].T @ (X[idx_k, :] @ mu[k, :])
            norm = np.linalg.norm(A_k_mu_k)
            mu[k, :] = A_k_mu_k / norm if norm > 1e-9 else X[np.random.randint(n), :]
            
    sims_sq = (X @ mu.T)**2
    max_sims = np.array([np.max(sims_sq[i, :]) for i in range(n)])
    objective = np.mean(max_sims)
    return labels, mu, sims_sq, objective

def run_diametrical_clustering(data: np.ndarray, K: int, n_init=50, max_iter=300, random_state=None):
    """High-level wrapper for running multiple replicates of diametrical clustering."""
    X = data.astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0; X /= norms
    rng_master = np.random.RandomState(random_state)
    best_objective = -np.inf
    best_result = {}
    for i in range(n_init):
        seed = rng_master.randint(np.iinfo(np.int32).max)
        labels, mu, sims_sq, obj = _diametrical_single_run(X, K, max_iter, seed)
        if obj > best_objective:
            best_objective, best_result = obj, {'labels': labels, 'centers': mu, 'sims_sq': sims_sq}
    labels, centers, sims_sq = best_result["labels"], best_result["centers"], best_result["sims_sq"]
    counts = np.bincount(labels, minlength=K); order = np.argsort(counts)[::-1]
    relabel_map = np.empty_like(order); relabel_map[order] = np.arange(K)
    return {
        "labels": relabel_map[labels],
        "centers": centers[order],
        "counts": counts[order],
        "distances": (1.0 - sims_sq[:, order]).astype(np.float32),
    }

# --- DATA LOADING & PATHING ---

def get_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Constructs input and output directories from command-line arguments."""
    # base_dir = Path(os.getenv("DATA_DIR", "./data"))
    base_dir = Path("/work3/s204684/SC2/data")
    print(f"[INFO] Using base data directory: {base_dir}")
    
    # Construct input path
    if args.data_type == 'source':
        input_dir = base_dir / "leading_source" / args.comparison_type / args.method / f"{args.freq_band}_{args.window_size}"
        output_dir = base_dir / "clustering_source" / args.cluster_method / args.comparison_type / args.method / f"{args.freq_band}_{args.window_size}"
    elif args.data_type == 'eeglab':
        input_dir = base_dir / "leading_eeg" / args.comparison_type / f"{args.freq_band}_{args.window_size}"
        output_dir = base_dir / "clustering_eeg" / args.cluster_method / args.comparison_type / f"{args.freq_band}_{args.window_size}"
    elif args.data_type == 'beamformer':
        input_dir = base_dir / "leading_source" / args.comparison_type / "beamformer" / f"{args.freq_band}_{args.window_size}"
        output_dir = base_dir / "clustering_source" / args.cluster_method / args.comparison_type / "beamformer" / f"{args.freq_band}_{args.window_size}"
    else:
        raise ValueError("Invalid data_type specified.")
    
    return input_dir, output_dir

def load_and_collate_data(input_dir: Path, conditions: List[str]) -> Tuple[np.ndarray, List, Dict]:
    """Loads all eigenvector .npy files, collates them, and builds metadata."""
    data_dict: Dict[str, Dict[str, np.ndarray]] = {}
    print(f"[LOAD] Searching for eigenvectors in: {input_dir.resolve()}")
    
    file_list = sorted(input_dir.glob("s_*-eigenvectors.npy"))
    if not file_list:
        raise FileNotFoundError(f"No eigenvector files found in {input_dir}")
        
    for f in file_list:
        parts = f.stem.split('_')
        subj, cond_raw = parts[1], parts[2].split('-')[0]
        if cond_raw in conditions:
            data_dict.setdefault(cond_raw, {})[subj] = np.load(f)

    print(f"[LOAD] Loaded data for {sum(len(v) for v in data_dict.values())} subject-condition files.")
    
    coll, meta = [], []
    for cond, subj_map in data_dict.items():
        for subj, eig in subj_map.items():
            n_epochs, n_windows, n_chan = eig.shape
            coll.append(eig.reshape(-1, n_chan))
            meta.extend((cond, subj, e, w) for e in range(n_epochs) for w in range(n_windows))
            
    coll_eigs = np.vstack(coll).astype(np.float32)
    print(f"[COLLATE] Total eigenvectors: {coll_eigs.shape[0]:,d} | Channels: {coll_eigs.shape[1]}")
    
    return coll_eigs, meta, data_dict

# --- MAIN DRIVER ---

def main():
    parser = argparse.ArgumentParser(
        description="Unified LEiDA clustering script (K-Means or Diametrical).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Clustering Method ---
    parser.add_argument('--cluster_method', type=str, required=True, choices=['kmeans', 'diametrical'],
                        help="The clustering algorithm to use.")
    # --- Path-building Arguments (must match eigenvector generation script) ---
    parser.add_argument('--data_type', type=str, required=True, choices=['source', 'eeglab', 'beamformer'],
                        help="The type of data that was processed to generate eigenvectors.")
    parser.add_argument('--comparison_type', type=str, default='all', help="Comparison type subdirectory.")
    parser.add_argument('--method', type=str, default='dSPM', 
                        help="Source reconstruction method (for 'source'/'beamformer' types).")
    parser.add_argument('--freq_band', type=str, default='alpha', help="Frequency band used.")
    parser.add_argument('--window_size', type=int, default=256, help="Window size used (in samples).")

    # --- Clustering Hyperparameters ---
    parser.add_argument('--k_min', type=int, default=2, help="Minimum K to test.")
    parser.add_argument('--k_max', type=int, default=20, help="Maximum K to test.")
    parser.add_argument('--n_init', type=int, default=50, help="Random restarts for each K.")
    parser.add_argument('--max_iter', type=int, default=200, help="Max iterations per restart.")
    parser.add_argument('--seed', type=int, default=42, help="Master random seed for reproducibility.")

    # --- Execution & I/O ---
    parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'])
    parser.add_argument('--cores', type=int, default=-1, help="CPU cores for parallelizing over K. -1 for all.")
    args = parser.parse_args()

    t0 = time.time()
    input_dir, output_dir = get_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- 1. Loading and Preparing Data ---")
    coll_eigs, meta, data_dict = load_and_collate_data(input_dir, args.conditions)
    
    # --- 2. Pre-process Data Based on Clustering Method ---
    if args.cluster_method == 'kmeans':
        processed_eigs = flip_eigenvectors(coll_eigs)
        cluster_func = run_kmeans_clustering
    else: # diametrical
        processed_eigs = coll_eigs # Normalization is handled inside the function
        cluster_func = run_diametrical_clustering

    # --- 3. Run Clustering in Parallel for each K ---
    ks = list(range(args.k_min, args.k_max + 1))
    n_cores = min(len(ks), os.cpu_count()) if args.cores == -1 else args.cores
    
    print(f"\n--- 2. Starting '{args.cluster_method}' clustering for K={ks[0]}..{ks[-1]} using {n_cores} cores ---")

    def _process_one_k(k):
        """Helper function to be called in parallel for each K."""
        results = cluster_func(
            processed_eigs, k,
            n_init=args.n_init, max_iter=args.max_iter, random_state=args.seed
        )
        labels_dict = map_labels_back(results["labels"], meta, data_dict)
        k_dir = output_dir / f"k_{k:02d}"; k_dir.mkdir(exist_ok=True)
        np.save(k_dir / "centers.npy", results["centers"])
        np.save(k_dir / "counts.npy", results["counts"])
        # Optional: distances can be very large. Comment out if not needed.
        np.save(k_dir / "distances.npy", results["distances"])
        with (k_dir / "labels.pkl").open("wb") as f:
            pickle.dump(labels_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return k

    Parallel(n_jobs=n_cores)(delayed(_process_one_k)(k) for k in ks)

    # --- 4. Finalize  ---
    print("\n--- 3. All Ks finished. Creating manifest file. ---")
    run_time = time.time() - t0

    print(f"Total execution time: {run_time:.2f} seconds.")
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()