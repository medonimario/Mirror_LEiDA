#!/usr/bin/env python3
"""
leida_batch_kmeans.py
---------------------

Run LEiDA-style K-means clustering for a *range* of K values (e.g. 4-20).

Outputs for each K:
    k_##/centers.npy         – cluster centroids  (K, n_channels)
    k_##/counts.npy          – cluster sizes      (K,)
    k_##/labels.pkl          – nested dict        {condition → subject → (n_epochs,n_windows)}
    k_##/distances.npy       – point-to-centroid distances (rows, K)   [optional, can be huge]
A run-level manifest.json stores all parameters & file hashes.

Requirements
------------
numpy, scikit-learn (≥0.24), joblib
"""
from __future__ import annotations
import argparse, json, os, pickle, time, hashlib, functools
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed, cpu_count

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


def run_leida_kmeans(data: np.ndarray, K: int, n_init=50, max_iter=200,
                     random_state=None):
    """K-means with clusters re-labelled by descending size."""
    km = KMeans(
        n_clusters=K, n_init=n_init, max_iter=max_iter,
        random_state=random_state, verbose=0
    ).fit(data)
    counts = np.bincount(km.labels_, minlength=K)
    order = np.argsort(counts)[::-1]      # biggest cluster first
    relabel = np.empty_like(order); relabel[order] = np.arange(K)
    return {
        "labels": relabel[km.labels_],
        "centers": km.cluster_centers_[order],
        "counts": counts[order],
        "distances": km.transform(data).astype(np.float32)   # large; delete if not needed
    }


def map_labels_back(labels_1d, meta, data_dict):
    """Back-project flat labels → dict[condition][subject] shape (epochs,windows)."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, (cond, subj, e, w) in zip(labels_1d, meta):
        tgt = out.setdefault(cond, {})\
                 .setdefault(subj, np.empty(data_dict[cond][subj].shape[:2], dtype=np.int16))
        tgt[e, w] = lab
    return out


# ---------------------------------------------------------------------
# 2) CLI & driver
# ---------------------------------------------------------------------
def sha1(path: Path, block=1<<20) -> str:         # tiny helper for manifest
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run K-means clustering for a range of K on LEiDA eigen-vectors."
    )
    p.add_argument("-i", "--input",  required=True, type=Path, help="directory with *_eigenvectors.npy")
    p.add_argument("-o", "--output", required=True, type=Path, help="where to save results")
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--cores", type=int, default=cpu_count(),
                   help="total CPU cores to use (<= physical cores)")
    p.add_argument("--n-init", type=int, default=50, help="k-means replicates per K")
    p.add_argument("--max-iter", type=int, default=200, help="k-means iterations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", type=str, nargs="+",
                   default=["Coordination", "Solo", "Spontaneous"],
                   help="conditions to load (default: all found)")
    return p.parse_args()


def save_one_k(outdir: Path, k: int, res: dict, labels_dict):
    """Persist results for a single K."""
    kd = outdir / f"k_{k:02d}"
    kd.mkdir(parents=True, exist_ok=True)
    np.save(kd / "centers.npy",   res["centers"])
    np.save(kd / "counts.npy",    res["counts"])
    # distances can be tens of GB; comment this line out if not needed
    np.save(kd / "distances.npy", res["distances"])
    with (kd / "labels.pkl").open("wb") as f:
        pickle.dump(labels_dict, f, protocol=4)
    print(f"[save]  K={k:<2d}  →  {kd}")


def main():
    args = parse_cli()
    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data_dict = load_leading_eigenvectors(args.input, args.conditions)
    print(f"[load]  {len(data_dict):d} conditions, {sum(len(v) for v in data_dict.values()):d} subjects")
    coll_eigs, meta = collate_eigenvectors(data_dict)

    ks = list(range(args.k_min, args.k_max + 1))
    n_jobs_inner = max(1, args.cores // len(ks))   # simple heuristic; tweak as desired

    # ---- parallel across K values ----
    def _process_k(k):
        res = run_leida_kmeans(coll_eigs, k, args.n_init, args.max_iter,
                               random_state=args.seed)
        labels_dict = map_labels_back(res["labels"], meta, data_dict)
        save_one_k(args.output, k, res, labels_dict)
        # free some RAM
        del res["distances"], res["labels"]
        return k

    Parallel(n_jobs=min(len(ks), args.cores))(delayed(_process_k)(k) for k in ks)

    # ---- manifest ----
    manifest = {
        "script": Path(__file__).name,
        "run_time_s": round(time.time() - t0, 2),
        "params": vars(args),
        "k_values": ks,
        "inputs_sha1": {f.name: sha1(f) for f in args.input.glob("*.npy")},
    }
    with (args.output / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done]  all Ks finished in {manifest['run_time_s']} s  "
          f"↗ see {args.output}")


if __name__ == "__main__":
    main()

#python src/mirror_leida/k-means.py --input data/leading_eeg_spontaneous/alpha --output data/kmeans/leading_eeg_spontaneous/alpha --k-min 4 --k-max 20