#!/usr/bin/env python3
"""
run_leida_diametrical_batch.py
------------------------------

Run **fast** diametrical clustering over a range of K values.
This version uses Numba to JIT-compile the core clustering loop,
achieving performance much closer to Scikit-learn's KMeans.

The methodology follows Olsen et al., NeuroImage (2022) and the
original MATLAB implementation.
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from joblib import Parallel, delayed, cpu_count

# Numba is the key to performance!
import numba

# Optional progress bar
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kwargs):
        # A minimal fallback if tqdm is not installed
        print(f"Running {kwargs.get('desc', 'replicates')} for {n} iterations...")
        return range(n)

# ---------------------------------------------------------------------
# 1. Data Loading & Preparation
# ---------------------------------------------------------------------

def load_leading_eigenvectors(data_dir: Path, possible_conditions=("Coordination", "Solo", "Spontaneous")) -> Dict[str, Dict[str, np.ndarray]]:
    """Load .npy eigen-vector files into a nested dict[condition][subject]."""
    data_dict: Dict[str, Dict[str, np.ndarray]] = {}
    print(f"[LOAD] Searching for .npy files in: {data_dir.resolve()}")
    
    file_list = sorted(data_dir.glob("*.npy"))
    if not file_list:
        raise FileNotFoundError(f"No .npy files found in the specified directory: {data_dir}")

    for f in file_list:
        if "eigenvectors" not in f.name:
            continue
        
        # Robust parsing for filenames like 's_101_Coordination-eigenvectors.npy'
        parts = f.stem.split('_')
        if len(parts) < 3:
            print(f"[LOAD]  ⚠︎ Skipping {f.name}: Unexpected filename format.")
            continue
        
        subj = parts[1]
        cond_raw = parts[2].split('-')[0]
        
        condition = next((c for c in possible_conditions if cond_raw.lower() == c.lower()), None)
        if condition is None:
            print(f"[LOAD]  ⚠︎ Skipping {f.name}: Unknown condition '{cond_raw}'.")
            continue

        arr = np.load(f)
        data_dict.setdefault(condition, {})[subj] = arr

    print(f"[LOAD] Loaded data for {sum(len(v) for v in data_dict.values())} subject-condition files.")
    return data_dict


def collate_eigenvectors(data_dict) -> Tuple[np.ndarray, List[Tuple[str, str, int, int]]]:
    """Stack all eigenvectors into a single 2D array and build a metadata list."""
    coll, meta = [], []
    for cond, subj_map in data_dict.items():
        for subj, eig in subj_map.items():
            n_epochs, n_windows, n_chan = eig.shape
            coll.append(eig.reshape(-1, n_chan))
            meta.extend((cond, subj, e, w) for e in range(n_epochs) for w in range(n_windows))
    
    # Use float32 to save memory, will be converted to float64 inside clustering for precision
    coll_eigs = np.vstack(coll).astype(np.float32)
    print(f"[COLLATE] Total eigenvectors: {coll_eigs.shape[0]:,d} | ROIs: {coll_eigs.shape[1]}")
    return coll_eigs, meta


# ---------------------------------------------------------------------
# 2. **FAST** Diametrical Clustering (Optimized with Numba)
# ---------------------------------------------------------------------

@numba.jit(nopython=True, fastmath=True, cache=True)
def _diametrical_single_run(X: np.ndarray, K: int, max_iter: int, seed: int):
    """A single, fast run of diametrical clustering, JIT-compiled by Numba."""
    n, p = X.shape
    np.random.seed(seed)

    # --- K-Means++ style seeding (no changes here) ---
    mu = np.empty((K, p), dtype=X.dtype)
    first_idx = np.random.randint(n)
    mu[0, :] = X[first_idx, :]
    
    min_sq_dists = np.full(n, np.inf, dtype=X.dtype)

    for i in range(1, K):
        dist_sq = 1.0 - (X @ mu[i-1, :].T)**2
        min_sq_dists = np.minimum(min_sq_dists, dist_sq)
        
        # Guard against sum being zero if all distances are zero
        s = min_sq_dists.sum()
        if s == 0.0:
            probs = np.full(n, 1.0/n, dtype=X.dtype)
        else:
            probs = min_sq_dists / s
        
        cum_probs = np.cumsum(probs)
        r = np.random.rand()
        next_idx = np.searchsorted(cum_probs, r)
        mu[i, :] = X[next_idx, :]

    # --- EM Iterations (no changes here) ---
    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(max_iter):
        sims_sq = (X @ mu.T)**2
        new_labels = np.argmax(sims_sq, axis=1).astype(np.int32)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        for k in range(K):
            idx_k = np.where(labels == k)[0]
            if len(idx_k) == 0:
                mu[k, :] = X[np.random.randint(n), :]
                continue
            
            A_k_mu_k = X[idx_k, :].T @ (X[idx_k, :] @ mu[k, :])
            norm = np.linalg.norm(A_k_mu_k)
            if norm > 1e-9:
                mu[k, :] = A_k_mu_k / norm
            else:
                mu[k, :] = X[np.random.randint(n), :]

    # --- Final objective calculation (THIS PART IS FIXED) ---
    sims_sq = (X @ mu.T)**2
    
    # Numba-compatible way to find max along an axis
    max_sims = np.empty(n, dtype=X.dtype)
    for i in range(n):
        max_sims[i] = np.max(sims_sq[i, :]) # Find max of each row
        
    objective = np.mean(max_sims)
    # --- END OF FIX ---

    return labels, mu, sims_sq, objective


def run_diametrical_clustering(data: np.ndarray, K: int, n_init: int = 50, max_iter: int = 200, random_state=None, verbose: bool = False):
    """High-level wrapper for running multiple replicates of diametrical clustering."""
    X = data.astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    rng_master = np.random.RandomState(random_state)
    best_objective = -np.inf
    best_result = {}

    rep_range = trange(n_init, desc=f"K={K:>2d}", disable=not verbose, leave=False, position=0)
    for _ in rep_range:
        seed = rng_master.randint(np.iinfo(np.int32).max)
        labels, mu, sims_sq, obj = _diametrical_single_run(X, K, max_iter, seed)
        
        if verbose:
            rep_range.set_postfix({"obj": f"{obj:.5f}"})

        if obj > best_objective:
            best_objective = obj
            best_result = {'labels': labels, 'centers': mu, 'sims_sq': sims_sq}

    labels = best_result["labels"]
    centers = best_result["centers"]
    sims_sq = best_result["sims_sq"]
    
    counts = np.bincount(labels, minlength=K)
    order = np.argsort(counts)[::-1]
    
    relabel_map = np.empty_like(order)
    relabel_map[order] = np.arange(K)

    return {
        "labels": relabel_map[labels],
        "centers": centers[order],
        "counts": counts[order],
        "distances": (1.0 - sims_sq[:, order]).astype(np.float32),
    }

# ---------------------------------------------------------------------
# 3. I/O and Helper functions (restored)
# ---------------------------------------------------------------------

def map_labels_back(labels_1d, meta, data_dict):
    """Reorganize 1D cluster labels back into a nested dictionary."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, (cond, subj, e, w) in zip(labels_1d, meta):
        if subj not in out.setdefault(cond, {}):
             out[cond][subj] = np.empty(data_dict[cond][subj].shape[:2], dtype=np.int16)
        out[cond][subj][e, w] = lab
    return out

def sha1(path: Path) -> str:
    """Calculates the SHA1 hash of a file for reproducibility."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):  # Read in 1MB chunks
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------
# 4. Command-Line Interface (restored)
# ---------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    """Sets up the command-line interface for the script."""
    p = argparse.ArgumentParser(
        description="Run FAST diametrical clustering over a range of K on LEiDA eigenvectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory containing *_eigenvectors.npy files")
    p.add_argument("-o", "--output", required=True, type=Path, help="Base directory to save clustering results")
    p.add_argument("--k-min", type=int, default=4, help="Minimum number of clusters (K) to test")
    p.add_argument("--k-max", type=int, default=20, help="Maximum number of clusters (K) to test")
    p.add_argument("--cores", type=int, default=cpu_count(), help="CPU cores for parallelizing over K values")
    p.add_argument("--n-init", type=int, default=50, help="Random restarts (replicates) for each K")
    p.add_argument("--max-iter", type=int, default=200, help="Max EM iterations per restart")
    p.add_argument("--seed", type=int, default=42, help="Master random seed for reproducibility")
    p.add_argument("-v", "--verbose", action="store_true", help="Show detailed progress bars (requires tqdm)")
    return p.parse_args()


# ---------------------------------------------------------------------
# 5. Main Driver
# ---------------------------------------------------------------------

def main():
    args = parse_cli()
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("--- 1. Loading data ---")
    data_dict = load_leading_eigenvectors(args.input)
    if not data_dict:
        print("Error: No data loaded. Exiting.")
        return
        
    coll_eigs, meta = collate_eigenvectors(data_dict)
    
    ks = list(range(args.k_min, args.k_max + 1))
    print(f"\n--- 2. Starting parallel clustering for K={ks[0]}..{ks[-1]} ---")

    def _process_one_k(k):
        """A helper function to be called in parallel for each K."""
        results = run_diametrical_clustering(
            coll_eigs, k,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.seed,
            verbose=args.verbose
        )
        labels_dict = map_labels_back(results["labels"], meta, data_dict)
        
        k_dir = args.output / f"k_{k:02d}"
        k_dir.mkdir(exist_ok=True)
        
        np.save(k_dir / "centers.npy", results["centers"])
        np.save(k_dir / "counts.npy", results["counts"])
        with (k_dir / "labels.pkl").open("wb") as f:
            pickle.dump(labels_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return k

    tasks = (delayed(_process_one_k)(k) for k in ks)
    Parallel(n_jobs=min(len(ks), args.cores))(tasks)

    print(f"\n--- 3. All Ks finished. Creating manifest file. ---")
    run_time = time.time() - t0

    # Create a manifest file for reproducibility
    manifest = {
        "script": Path(__file__).name,
        "run_time_seconds": round(run_time, 2),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "k_values_processed": ks,
        "input_data_sha1": {f.name: sha1(f) for f in args.input.glob("*.npy") if "eigenvectors" in f.name},
    }
    with (args.output / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        
    print(f"Total execution time: {run_time:.2f} seconds.")
    print(f"Results saved in: {args.output.resolve()}")


if __name__ == "__main__":
    main()