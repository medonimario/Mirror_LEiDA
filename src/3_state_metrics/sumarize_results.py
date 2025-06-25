#!/usr/bin/env python3
"""
summarize_results.py
--------------------

Scans the clustering results directories to provide a quick overview of
which analyses yielded statistically significant findings.

It searches for 'STATS_GLOBAL_FO.csv' and 'STATS_GLOBAL_DT.csv' files,
checks them for significant results (p_fdr_global < 0.05), and prints
a color-coded tree with the following symbols:

- ✅ : At least one significant finding.
- ❌ : No significant findings.
- ❔ : File not found or could not be read.
"""
import os
import pandas as pd
from pathlib import Path
import argparse

# --- ANSI Color Codes for Terminal Output ---
class Colors:
    GREEN = '\033[92m'  # Green for checkmark
    RED = '\033[91m'    # Red for cross
    YELLOW = '\033[93m' # Yellow for question mark
    BLUE = '\033[94m'   # Blue for directory names
    ENDC = '\033[0m'    # Resets the color

# --- Core Functions ---

def check_csv_for_significance(file_path: Path) -> str:
    """
    Reads a stats CSV and returns a symbol based on its contents.
    """
    if not file_path.exists():
        return f"{Colors.YELLOW}❔{Colors.ENDC}"
    
    try:
        df = pd.read_csv(file_path)
        if 'significant_fdr_global' not in df.columns:
            return f"{Colors.YELLOW}❔{Colors.ENDC}"
        
        # .any() returns True if at least one value in the series is True
        if df['significant_fdr_global'].any():
            return f"{Colors.GREEN}✅{Colors.ENDC}"
        else:
            return f"{Colors.RED}❌{Colors.ENDC}"
    except Exception:
        # Handle empty files or other pandas errors
        return f"{Colors.YELLOW}❔{Colors.ENDC}"

def build_results_tree(base_dir: Path) -> dict:
    """
    Traverses the directory structure and builds a nested dictionary of results.
    """
    results_tree = {}
    
    # Find all unique directories that contain our statistics files
    analysis_dirs = sorted(list(set(p.parent for p in base_dir.rglob("STATS_GLOBAL_*.csv"))))

    for dir_path in analysis_dirs:
        # Get the relative path to build the tree structure
        relative_parts = dir_path.relative_to(base_dir).parts
        
        # Check status for both FO and DT files
        fo_status = check_csv_for_significance(dir_path / "STATS_GLOBAL_FO_perm.csv")
        dt_status = check_csv_for_significance(dir_path / "STATS_GLOBAL_DT_perm.csv")
        
        # Navigate or create the nested dictionary structure
        current_level = results_tree
        for part in relative_parts:
            # setdefault is perfect for creating nested dicts on the fly
            current_level = current_level.setdefault(part, {})
            
        # Store the final result tuple at the leaf
        current_level['results'] = (fo_status, dt_status)

    return results_tree

def print_tree(tree_dict: dict, indent: str = ""):
    """
    Recursively prints the nested dictionary as a formatted tree.
    """
    # Sort keys to ensure consistent order, putting 'results' last
    items = sorted(tree_dict.items(), key=lambda x: x[0] == 'results')

    for i, (key, value) in enumerate(items):
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        
        if key == 'results':
            # This is a leaf node with the results tuple
            fo_status, dt_status = value
            print(f"{indent}{prefix}[{fo_status} FO] [{dt_status} DT]")
        elif isinstance(value, dict):
            # This is a directory node
            print(f"{indent}{prefix}{Colors.BLUE}{key}{Colors.ENDC}")
            new_indent = indent + ("    " if is_last else "│   ")
            print_tree(value, new_indent)

# --- Main Driver ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary tree of LEiDA statistical results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--base_dir', 
        type=Path, 
        default=Path(os.getenv("DATA_DIR", "./data")),
        help="The base data directory containing 'clustering_eeg' and 'clustering_source'."
    )
    args = parser.parse_args()

    print(f"Scanning for results under: {args.base_dir.resolve()}\n")
    
    # Process both main data types
    for data_type_root in ["clustering_eeg", "clustering_source"]:
        root_path = args.base_dir / data_type_root
        if root_path.exists():
            print(f"{Colors.BLUE}{'='*15} {data_type_root.upper()} {'='*15}{Colors.ENDC}")
            results_tree = build_results_tree(root_path)
            print_tree(results_tree)
            print("\n")
        else:
            print(f"{Colors.YELLOW}Directory not found, skipping: {root_path}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()