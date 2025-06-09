#!/usr/bin/env python

import os
import argparse
from dotenv import load_dotenv
from joblib import Parallel, delayed
import numpy as np
import mne
import traceback
import sys
from datetime import datetime

# Import your LEiDAEEGAnalyzer class
# from leida_eeg_analyzer import LEiDAEEGAnalyzer
from leida_eeg_analyzer_patched import LEiDAEEGAnalyzer

# Global variable for our "omitted.log" path
omitted_log_path = None

# ----------------------------------------------------------
# Example usage:
#   python leading_subjects.py --freq_band alpha --window_size 32
# ----------------------------------------------------------

def log_message(log_file, msg):
    """Simple helper to write a string to a log file and to stdout."""
    with open(log_file, 'a') as lf:
        lf.write(msg + "\n")
    print(msg)

def process_subject(ppt, pair_number, condition,
                    source_path, leading_path, freq_band, window_size,
                    remove_edges, do_plots, verbose):

    """Compute LEiDA leading eigenvectors for a single subject/condition."""
    omitted_log_path = os.path.join(leading_path, 'omitted.log')
    # s_number: The numeric subject code (101..144 for PPT1, 201..244 for PPT2)
    if ppt == 'PPT1':
        s_number = str(100 + pair_number)
        source_path = os.path.join(source_path, 'PPT1')
    else:
        s_number = str(200 + pair_number)
        source_path = os.path.join(source_path, 'PPT2')

    # Example of a subject-level log path
    out_dir = os.path.join(leading_path, freq_band)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"s_{s_number}_{condition}.log")

    log_message(log_file, f"===== LEiDA Processing Start =====")
    log_message(log_file, f"Subject: {ppt}, {s_number}, Condition: {condition}")
    log_message(log_file, f"Band: {freq_band}, Window size: {window_size}, remove_edges={remove_edges}")

    # 1) Check existence of the source file
    epo_fname = os.path.join(source_path, f"s_{s_number}_{condition}.set")
    if not os.path.exists(epo_fname):
        print(f"Skipped missing file: {epo_fname}")
        if omitted_log_path:
            with open(omitted_log_path, 'a') as lf:
                lf.write(f"{ppt} {s_number} {condition}\n")
        log_message(log_file, f"File {epo_fname} does not exist; skipping.")
        log_message(log_file, "===== LEiDA Processing End =====\n")
        return

    try:
        # --------------------------------------------------
        # 1) Load epochs
        # --------------------------------------------------
        log_message(log_file, f"Loading {epo_fname}")
        epochs = mne.io.read_epochs_eeglab(epo_fname, verbose='ERROR')
        epochs.crop(tmin=0.0)
        log_message(log_file, f"Empochs tmin: {epochs.tmin}, tmax: {epochs.tmax}")
        data_3d = epochs.get_data()  # shape: (n_epochs, n_ROIs, n_timepoints)
        fs = epochs.info['sfreq']
        # window_size = int(fs / 8)

        log_message(log_file, f"Data loaded. Shape = {data_3d.shape}, sfreq={fs}")

        # --------------------------------------------------
        # 2) Instantiate LEiDAEEGAnalyzer
        # --------------------------------------------------
        analyzer = LEiDAEEGAnalyzer(fs=fs,
                                    freq_band=freq_band,
                                    window_size=window_size,
                                    remove_edges=remove_edges,
                                    do_plots=do_plots,
                                    verbose=verbose)

        # --------------------------------------------------
        # 3) Compute the leading eigenvectors
        # --------------------------------------------------
        all_eigenvectors = analyzer.compute_leading_eigenvectors(data_3d)
        log_message(log_file, f"Computed leading eigenvectors. Shape={all_eigenvectors.shape} (epochs x windows x channels)")

        # --------------------------------------------------
        # 4) Save the result
        # --------------------------------------------------
        # E.g. data/leading/dSPM/alpha/s_101_Coordination-eigenvectors.npy
        eigen_fname = os.path.join(out_dir, f"s_{s_number}_{condition}-eigenvectors.npy")
        np.save(eigen_fname, all_eigenvectors)
        log_message(log_file, f"Saved eigenvectors to {eigen_fname}")

    except Exception as e:
        # Write the error to both the log file and the console
        err_str = "ERROR: " + str(e) + "\n" + traceback.format_exc()
        log_message(log_file, err_str)

    finally:
        log_message(log_file, "===== LEiDA Processing End =====")
        log_message(log_file, "")  # blank line for clarity


def main():
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "./data")
    # source_path = os.path.join(data_dir, "raw_spontaneous")
    # leading_path = os.path.join(data_dir, "leading_eeg_spontaneousW1")
    source_path = os.path.join(data_dir, "raw")
    leading_path = os.path.join(data_dir, "leading_eegW1_D")

    # ------------------------------------------
    # Command-line arguments
    # ------------------------------------------
    parser = argparse.ArgumentParser(description="Step2 LEiDA computation script")
    parser.add_argument('--freq_band', type=str, default='alpha', 
                        help="Frequency band: 'alpha', 'beta', 'gamma', etc.")
    parser.add_argument('--window_size', type=int, default=256, # 32,
                        help="Size of each window in samples for dynamic PL calculation")
    parser.add_argument('--remove_edges', type=bool, default=False,
                        help="If set, skip the first and last window in each epoch for the LEiDA analysis.")
    parser.add_argument('--do_plots', type=bool, default=False,
                        help="Enable diagnostic plots inside LEiDAEEGAnalyzer.")
    parser.add_argument('--verbose', type=bool, default=True,
                        help="Enable verbose output in LEiDAEEGAnalyzer.")
    parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'],)
    args = parser.parse_args()

    # Prepare output directory
    out_dir = os.path.join(leading_path, args.freq_band)
    os.makedirs(out_dir, exist_ok=True)

    ppts = ['PPT1', 'PPT2']
    pair_numbers = list(range(1, 44))  # [1..43]
    conditions = args.conditions

    # Parallel call: process each combination
    Parallel(n_jobs=-1)(
        delayed(process_subject)(
            ppt, pn, cond,
            source_path=source_path,
            leading_path=leading_path,
            freq_band=args.freq_band,
            window_size=args.window_size,
            remove_edges=args.remove_edges,
            do_plots=args.do_plots,
            verbose=args.verbose
        )
        for ppt in ppts
        for pn in pair_numbers
        for cond in conditions
    )


if __name__ == "__main__":
    main()
