import os
import argparse
from dotenv import load_dotenv
from joblib import Parallel, delayed
import numpy as np
import mne
import traceback
import matplotlib.pyplot as plt

# Import your LEiDAEEGAnalyzer class
# Make sure the file is named 'leida_eeg_analyzer.py' or adjust the import
from leida_eeg_analyzer import LEiDAEEGAnalyzer

def process_subject(
    ppt, 
    pair_number, 
    condition,
    source_data_dir, # Directory with beamformer outputs
    leida_output_dir,  # Directory to save eigenvector .npy files
    freq_band, 
    window_size,
    remove_edges,
    verbose
):
    """
    Computes LEiDA leading eigenvectors for a single source-reconstructed subject file.
    """
    s_number = str(100 + pair_number) if ppt == 'PPT1' else str(200 + pair_number)
    file_identifier = f"s_{s_number}_{condition}"

    # --- 1. CHECKPOINTING: Define output and check if it exists ---
    # The output path is now the primary check.
    output_fname = os.path.join(leida_output_dir, f"{file_identifier}-eigenvectors.npy")
    if os.path.exists(output_fname):
        # Quietly skip if already done to not clutter the log
        # print(f"Output exists, skipping: {output_fname}")
        return

    # --- 2. Define input and check if it exists ---
    # Input file is now a .fif file from the beamformer pipeline
    input_fname = os.path.join(source_data_dir, f"{file_identifier}-source-beamformer-epo.fif")
    if not os.path.exists(input_fname):
        print(f"Skipping missing source file: {input_fname}")
        # Optional: Log to an omitted file if desired
        return

    try:
        print(f"Processing: {file_identifier} | Freq: {freq_band} | WinSize: {window_size}")
        
        # --- 3. Load the source-level epochs file ---
        # Use mne.read_epochs for .fif files
        epochs = mne.read_epochs(input_fname, verbose='ERROR')
        data_3d = epochs.get_data()  # shape: (n_epochs, n_ROIs, n_timepoints)
        fs = epochs.info['sfreq']
        
        # --- 4. Instantiate and run the LEiDA Analyzer ---
        # The analyzer works on the numpy array directly, no changes needed here.
        analyzer = LEiDAEEGAnalyzer(
            fs=fs,
            freq_band=freq_band,
            window_size=window_size,
            remove_edges=remove_edges,
            do_plots=False,  # Turn off plotting for batch jobs
            verbose=verbose
        )
        all_eigenvectors = analyzer.compute_leading_eigenvectors(data_3d)

        # --- 5. Save the result ---
        np.save(output_fname, all_eigenvectors)
        print(f"--> Saved eigenvectors to {output_fname} | Shape: {all_eigenvectors.shape}")

    except Exception as e:
        print(f"---! ERROR processing {file_identifier}: {e} !---")
        traceback.print_exc()
    
    finally:
        # Prevent memory leaks from any potential plots
        plt.close('all')


def main():
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "./data")
    
    # Define clear input and output paths
    source_data_dir = os.path.join(data_dir, "source", "beamformer")
    leida_output_dir_base = os.path.join(data_dir, "leading_beam")

    parser = argparse.ArgumentParser(description="LEiDA computation on source-reconstructed data")
    parser.add_argument('--freq_band', type=str, default='alpha', help="Frequency band: 'alpha', 'beta', 'gamma'")
    parser.add_argument('--window_size', type=int, default=256, help="Window size in samples for dPL calculation")
    parser.add_argument('--remove_edges', action='store_true', help="If set, skip the first and last window in each epoch.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output in LEiDAEEGAnalyzer.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs.")
    args = parser.parse_args()

    # Create a specific output directory for this analysis run
    leida_output_dir = os.path.join(leida_output_dir_base, f"freq_{args.freq_band}_win_{args.window_size}")
    os.makedirs(leida_output_dir, exist_ok=True)
    print(f"Saving LEiDA results to: {leida_output_dir}")

    # Define subjects and conditions
    ppts = ['PPT1', 'PPT2']
    pair_numbers = list(range(1, 44))
    conditions = ['Coordination', 'Solo', 'Spontaneous']

    # Use a generator expression for memory-safe parallelization
    tasks = (
        delayed(process_subject)(
            ppt, pn, cond,
            source_data_dir=source_data_dir,
            leida_output_dir=leida_output_dir,
            freq_band=args.freq_band,
            window_size=args.window_size,
            remove_edges=args.remove_edges,
            verbose=args.verbose
        )
        for ppt in ppts
        for pn in pair_numbers
        for cond in conditions
    )
    
    total_tasks = len(ppts) * len(pair_numbers) * len(conditions)
    print(f"\nStarting LEiDA processing for {total_tasks} files...")
    print("Already completed files will be skipped automatically.\n")

    Parallel(n_jobs=args.n_jobs)(tasks)
    
    print("\nLEiDA batch processing complete.")


if __name__ == "__main__":
    main()