#!/usr/bin/env python

import os
import argparse
from dotenv import load_dotenv
from joblib import Parallel, delayed
import numpy as np
import mne
import traceback
import matplotlib.pyplot as plt

from leida_eeg_analyzer import LEiDAEEGAnalyzer

# --- CONFIGURATION HELPER ---

def get_data_config(data_type, base_dir, comparison_type,  method, freq_band, window_size):
    """
    Returns a configuration dictionary with paths and functions based on data type.
    
    This function isolates all the details that differ between data types.
    """
    config = {}
    
    if data_type == 'source':
        # For standard source reconstruction files (.fif)
        input_dir = os.path.join(base_dir, "source", f"source_{comparison_type}" ,method)
        output_dir = os.path.join(base_dir, "leading_source", f"{comparison_type}", method, f"{freq_band}_{window_size}")
        
        config['input_template'] = os.path.join(input_dir, "s_{s_number}_{condition}-source.fif")
        config['output_template'] = os.path.join(output_dir, "s_{s_number}_{condition}-eigenvectors.npy")
        config['load_function'] = mne.read_epochs
        config['extra_processing_fn'] = None # No extra steps needed
        
    elif data_type == 'eeglab':
        # For raw sensor-space data from EEGLAB (.set)
        # Note: This assumes a PPT1/PPT2 subdirectory structure for inputs.
        # The output path is simplified to be consistent.
        input_dir_ppt1 = os.path.join(base_dir, "raw_eeg", f"raw_{comparison_type}", "PPT1")
        input_dir_ppt2 = os.path.join(base_dir, "raw_eeg", f"raw_{comparison_type}", "PPT2")
        output_dir = os.path.join(base_dir, "leading_eeg", f"{comparison_type}", f"{freq_band}_{window_size}")
        
        config['input_template'] = {
            'PPT1': os.path.join(input_dir_ppt1, "s_{s_number}_{condition}.set"),
            'PPT2': os.path.join(input_dir_ppt2, "s_{s_number}_{condition}.set")
        }
        config['output_template'] = os.path.join(output_dir, "s_{s_number}_{condition}-eigenvectors.npy")
        config['load_function'] = mne.io.read_epochs_eeglab
        # Define the extra cropping step as a lambda function
        config['extra_processing_fn'] = lambda epochs: epochs.crop(tmin=0.0)
        
    elif data_type == 'beamformer':
        # For beamformer source reconstruction files (.fif)
        input_dir = os.path.join(base_dir, "source", f"source_{comparison_type}", "beamformer")
        output_dir = os.path.join(base_dir, "leading_source", f"{comparison_type}", "beamformer" ,f"{freq_band}_{window_size}")

        config['input_template'] = os.path.join(input_dir, "s_{s_number}_{condition}-source-beamformer-epo.fif")
        config['output_template'] = os.path.join(output_dir, "s_{s_number}_{condition}-eigenvectors.npy")
        config['load_function'] = mne.read_epochs
        config['extra_processing_fn'] = None

    else:
        raise ValueError(f"Unknown data_type: {data_type}. Choose from 'source', 'eeglab', 'beamformer'.")

    # Ensure output directory exists for this run
    os.makedirs(os.path.dirname(config['output_template']), exist_ok=True)
    return config


# --- CORE PROCESSING FUNCTION ---

def process_subject(
    ppt, 
    pair_number, 
    condition,
    config,
    analyzer_params
):
    """
    Generic function to compute LEiDA leading eigenvectors for a single subject.
    It uses the provided config dictionary to handle different data types.
    """
    s_number = str(100 + pair_number) if ppt == 'PPT1' else str(200 + pair_number)
    
    # 1. Construct file paths using templates from the config
    # The format_map allows using placeholders directly from a dict.
    format_dict = {
        's_number': s_number,
        'condition': condition,
        'ppt': ppt
    }
    
    # Handle potentially dict-based input templates (like for EEGLAB)
    if isinstance(config['input_template'], dict):
        input_fname = config['input_template'][ppt].format_map(format_dict)
    else:
        input_fname = config['input_template'].format_map(format_dict)
        
    output_fname = config['output_template'].format_map(format_dict)

    # 2. Checkpointing: Skip if the OUTPUT file already exists
    if os.path.exists(output_fname):
        return f"Skipped (already exists): {os.path.basename(output_fname)}"

    # 3. Check for input file existence
    if not os.path.exists(input_fname):
        return f"Skipped (input missing): {os.path.basename(input_fname)}"

    try:
        # 4. Load data using the specified loading function
        epochs = config['load_function'](input_fname, verbose='ERROR')
        
        # 5. Apply any extra processing steps if defined
        if config['extra_processing_fn']:
            epochs = config['extra_processing_fn'](epochs)

        data_3d = epochs.get_data()
        fs = epochs.info['sfreq']
        
        # 6. Instantiate and run the LEiDA Analyzer
        analyzer = LEiDAEEGAnalyzer(fs=fs, **analyzer_params)
        all_eigenvectors = analyzer.compute_leading_eigenvectors(data_3d)

        # 7. Save the result
        np.save(output_fname, all_eigenvectors)
        return f"Success: {os.path.basename(output_fname)} | Shape: {all_eigenvectors.shape}"

    except Exception as e:
        error_msg = f"ERROR on {os.path.basename(input_fname)}: {e}"
        # For debugging, uncomment the line below
        # traceback.print_exc()
        return error_msg
    
    finally:
        # Prevent memory leaks from any potential plots if do_plots=True
        plt.close('all')


# --- MAIN EXECUTION BLOCK ---

def main():
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "./data")
    
    parser = argparse.ArgumentParser(description="Unified batch processing script for LEiDA analysis.")
    
    # --- Primary arguments ---
    parser.add_argument('--data_type', type=str, required=True, 
                        choices=['source', 'eeglab', 'beamformer'],
                        help="The type of data to process.")
    parser.add_argument('--freq_band', type=str, default='alpha', 
                        help="Frequency band: 'alpha', 'beta', or 'gamma'.")
    parser.add_argument('--window_size', type=int, default=256,
                        help="Size of the analysis window in samples.")
    parser.add_argument('--comparison_type', type=str, default='all')
    parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'],
                        help="List of conditions to process. Default: ['Coordination', 'Solo', 'Spontaneous']")
    
    # --- Optional arguments ---
    parser.add_argument('--method', type=str, default='dSPM', 
                        help="Source reconstruction method (used for 'source' data_type). E.g., dSPM, MNE.")
    parser.add_argument('--remove_edges', action='store_true',
                        help="If set, skip the first and last window in each epoch.")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose output from the LEiDA analyzer class.")
    parser.add_argument('--n_jobs', type=int, default=-1, 
                        help="Number of parallel jobs to run (-1 uses all available cores).")
    args = parser.parse_args()

    # Get the specific configuration for this run
    config = get_data_config(
        data_type=args.data_type,
        base_dir=data_dir,
        comparison_type=args.comparison_type,
        method=args.method,
        freq_band=args.freq_band,
        window_size=args.window_size
    )

    # Bundle analyzer parameters into a dictionary for clean passing
    analyzer_params = {
        'freq_band': args.freq_band,
        'window_size': args.window_size,
        'remove_edges': args.remove_edges,
        'do_plots': False,  # Plots should be off for batch processing
        'verbose': args.verbose
    }

    # Define subject and condition loops
    ppts = ['PPT1', 'PPT2']
    pair_numbers = list(range(1, 44))
    conditions = args.conditions

    tasks = (
        delayed(process_subject)(
            ppt, pn, cond,
            config=config,
            analyzer_params=analyzer_params
        )
        for ppt in ppts
        for pn in pair_numbers
        for cond in conditions
    )
    
    total_tasks = len(ppts) * len(pair_numbers) * len(conditions)
    print(f"--- Starting LEiDA Batch Processing ---")
    print(f"Data Type: {args.data_type.upper()} | Freq: {args.freq_band} | WinSize: {args.window_size}")
    print(f"Total tasks to check: {total_tasks}")
    print(f"Output directory: {os.path.dirname(config['output_template'])}")
    print(f"Running with {args.n_jobs if args.n_jobs != -1 else 'all'} cores...\n")

    results = Parallel(n_jobs=args.n_jobs)(tasks)
    
    # Print a summary of the results
    success_count = sum(1 for r in results if r and r.startswith('Success'))
    skipped_exist = sum(1 for r in results if r and 'already exists' in r)
    skipped_missing = sum(1 for r in results if r and 'input missing' in r)
    error_count = sum(1 for r in results if r and r.startswith('ERROR'))

    print("\n--- Batch Processing Complete ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_exist}")
    print(f"Skipped (input missing): {skipped_missing}")
    print(f"Errors: {error_count}")
    if error_count > 0:
        print("\nErrors encountered:")
        for r in results:
            if r and r.startswith('ERROR'):
                print(f" - {r}")


if __name__ == "__main__":
    main()