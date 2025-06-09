#!/usr/bin/env python3
"""
leida_batch_diametrical.py
--------------------------

Run **diametrical** clustering (Sra & Karp 2012, Algorithm 2) for a *range* of K values (e.g. 4-20).

The command-line interface, folder layout and outputs are **identical** to the original
`leida_batch_kmeans.py` - except for two additions:

* **Progress reporting**: use `--verbose` (or `-v`) to see a live progress bar for each
  replicate within every K (powered by *tqdm* when available, else falls back to simple prints).
* **Distances** are now stored as `1 - cos²(·)` (orientation-invariant) instead of Euclidean.

Outputs for each K (unchanged):
    k_##/centers.npy         - cluster centroids  (K, n_channels)
    k_##/counts.npy          - cluster sizes      (K,)
    k_##/labels.pkl          - nested dict        {condition → subject → (n_epochs,n_windows)}
    k_##/distances.npy       - 1 - cos² distances (rows, K)   [optional, can be huge]
A run-level manifest.json stores all parameters & file hashes.

Requirements
------------
Only numpy & joblib are mandatory.  *tqdm* (for fancy progress bars) is optional.
`scikit-learn` is kept as an import for backward compatibility but **not used**.
"""
from __future__ import annotations
import argparse, json, pickle, time, hashlib, sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from sklearn.cluster import KMeans        # kept for compatibility; NOT used
from joblib import Parallel, delayed, cpu_count

# optional progress bar ------------------------------------------------
try:
    from tqdm import trange
except ImportError:  # minimal fallback
    def trange(n, **kwargs):
        return range(n)

# ---------------------------------------------------------------------
# 1) helpers copied (lightly cleaned) from the notebook
# ---------------------------------------------------------------------

def load_leading_eigenvectors(data_dir: Path, possible_conditions=("Coordination", "Solo", "Spontaneous")) -> Dict[str, Dict[str, np.ndarray]]:
    """Load .npy eigen-vector files → nested dict[condition][subject]."""
    data_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(data_dir.glob("*.npy")):
        if "eigenvectors" not in f.name:
            continue
        base = f.stem[2:] if f.stem.startswith("s_") else f.stem
        subj, rest = base.split("_", 1)
        cond_raw = rest.split("-")[0]
        condition = next((c for c in possible_conditions if cond_raw.lower().startswith(c.lower())), None)
        if condition is None:
            print(f"[load]  ⚠︎ skipping {f.name}: unknown condition")
            continue
        arr = np.load(f)
        data_dict.setdefault(condition, {})[subj] = arr
        print(f"[load]  {f.name:40s}  →  {condition}/{subj}  {arr.shape}")
    return data_dict


def collate_eigenvectors(data_dict) -> Tuple[np.ndarray, List[Tuple[str, str, int, int]]]:
    """Stack all (epochs,windows,channels) → (rows,channels) and build meta list."""
    coll, meta = [], []
    for cond, subj_map in data_dict.items():
        for subj, eig in subj_map.items():
            n_epochs, n_windows, n_chan = eig.shape
            coll.append(eig.reshape(-1, n_chan))
            meta.extend((cond, subj, e, w) for e in range(n_epochs) for w in range(n_windows))
    coll_eigs = np.vstack(coll).astype(np.float32)
    print(f"[collate] total rows: {coll_eigs.shape[0]:,d}   channels: {coll_eigs.shape[1]}")
    return coll_eigs, meta


# ---------------------------------------------------------------------
# 2) **NEW** – diametrical clustering implementation
# ---------------------------------------------------------------------

def _diametrical_single_run(X: np.ndarray, K: int, max_iter: int, rng: np.random.RandomState, verbose: bool = False):
    """One replicate of diametrical clustering.

    Parameters
    ----------
    X : (n, p) unit-norm rows.
    K : int, number of clusters.
    max_iter : int, EM iterations.
    rng : RandomState for reproducibility.
    verbose : bool, progress printing.
    """
    n, p = X.shape

    # --- ++-style seeding -------------------------------------------
    first = rng.randint(n)
    mu = [X[first]]
    for _ in range(1, K):
        sims = np.square(X @ np.vstack(mu).T)           # (n,len(mu))
        best_sim = sims.max(axis=1)
        probs = best_sim.max() - best_sim + 1e-9        # choose far points
        probs /= probs.sum()
        mu.append(X[rng.choice(n, p=probs)])
    mu = np.vstack(mu)

    labels = np.full(n, -1, np.int32)

    it_range = trange(max_iter, disable=not verbose, leave=False, desc="EM", position=1)
    for it in it_range:
        # E-step ------------------------------------------------------
        sims = np.square(X @ mu.T)                      # (n,K)
        new_labels = sims.argmax(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # M-step ------------------------------------------------------
        for k in range(K):
            idx = labels == k
            if not idx.any():
                mu[k] = X[rng.randint(n)]
                continue
            A = X[idx].T @ X[idx]
            v = A @ mu[k]
            nv = np.linalg.norm(v)
            mu[k] = v / nv if nv > 0 else X[rng.randint(n)]

    sims = np.square(X @ mu.T)
    objective = sims.max(axis=1).mean()
    return labels, mu, sims, objective


def run_leida_kmeans(data: np.ndarray, K: int, n_init: int = 50, max_iter: int = 200, random_state=None, verbose: bool = False):
    """Drop-in replacement for the old K-means routine - now diametrical clustering."""
    X = data.astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms                                           # unit-norm rows

    rng_master = np.random.RandomState(random_state)
    best = {"obj": -np.inf}

    rep_range = trange(n_init, desc=f"K={K:>2d}", disable=not verbose, position=0)
    for rep in rep_range:
        rng = np.random.RandomState(rng_master.randint(2 ** 32))
        labels, mu, sims, obj = _diametrical_single_run(X, K, max_iter, rng, verbose=verbose)
        rep_range.set_postfix({"obj": f"{obj:.4f}"})
        if obj > best.get("obj", -np.inf):
            best.update(obj=obj, labels=labels, centers=mu, sims=sims)

    labels = best["labels"]
    centers = best["centers"]
    sims = best["sims"]

    # reorder clusters by descending size ---------------------------
    counts = np.bincount(labels, minlength=K)
    order = np.argsort(counts)[::-1]
    relabel = np.empty_like(order)
    relabel[order] = np.arange(K)

    return {
        "labels": relabel[labels],
        "centers": centers[order],
        "counts": counts[order],
        "distances": (1.0 - sims[:, order]).astype(np.float32),
    }


# ---------------------------------------------------------------------
# 3) label mapping (unchanged) & I/O helpers
# ---------------------------------------------------------------------

def map_labels_back(labels_1d, meta, data_dict):
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, (cond, subj, e, w) in zip(labels_1d, meta):
        tgt = out.setdefault(cond, {}).setdefault(
            subj, np.empty(data_dict[cond][subj].shape[:2], dtype=np.int16)
        )
        tgt[e, w] = lab
    return out


def sha1(path: Path, block=1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------
# 4) CLI driver
# ---------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run *diametrical* clustering (with progress bars) over a range of K on LEiDA eigen-vectors."
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="directory with *_eigenvectors.npy")
    p.add_argument("-o", "--output", required=True, type=Path, help="where to save results")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--cores", type=int, default=cpu_count(), help="total CPU cores to use (<= physical cores)")
    p.add_argument("--n-init", type=int, default=50, help="replicates per K (random restarts)")
    p.add_argument("--max-iter", type=int, default=200, help="EM iterations per replicate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", type=str, nargs="+", default=["Coordination", "Solo", "Spontaneous"], help="conditions to load (default: all found)")
    p.add_argument("-v", "--verbose", action="store_true", help="show progress bars (requires tqdm)")
    return p.parse_args()


def save_one_k(outdir: Path, k: int, res: dict, labels_dict):
    kd = outdir / f"k_{k:02d}"
    kd.mkdir(parents=True, exist_ok=True)
    np.save(kd / "centers.npy", res["centers"])
    np.save(kd / "counts.npy", res["counts"])
    np.save(kd / "distances.npy", res["distances"])
    with (kd / "labels.pkl").open("wb") as f:
        pickle.dump(labels_dict, f, protocol=4)
    print(f"[save]  K={k:<2d}  →  {kd}")


# ---------------------------------------------------------------------
# 5) main driver – calls the *new* clustering function
# ---------------------------------------------------------------------

def main():
    args = parse_cli()
    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data_dict = load_leading_eigenvectors(args.input, args.conditions)
    print(f"[load]  {len(data_dict):d} conditions, {sum(len(v) for v in data_dict.values()):d} subjects")
    coll_eigs, meta = collate_eigenvectors(data_dict)

    ks = list(range(args.k_min, args.k_max + 1))

    # ---- parallel across K values ---------------------------------
    def _process_k(k):
        res = run_leida_kmeans(
            coll_eigs,
            k,
            args.n_init,
            args.max_iter,
            random_state=args.seed,
            verbose=args.verbose,
        )
        labels_dict = map_labels_back(res["labels"], meta, data_dict)
        save_one_k(args.output, k, res, labels_dict)
        del res["distances"], res["labels"]  # free RAM
        return k

    Parallel(n_jobs=min(len(ks), args.cores))(delayed(_process_k)(k) for k in ks)

    # ---- manifest --------------------------------------------------
    manifest = {
        "script": Path(__file__).name,
        "run_time_s": round(time.time() - t0, 2),
        "params": vars(args),
        "k_values": ks,
        "inputs_sha1": {f.name: sha1(f) for f in args.input.glob("*.npy")},
    }
    with (args.output / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done]  all Ks finished in {manifest['run_time_s']} s  ↗ see {args.output}")


if __name__ == "__main__":
    main()
