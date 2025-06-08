import os
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from joblib import Parallel, delayed

from source_recon_beamformer import EEGBeamformerSourceReconstruction

load_dotenv()
data_dir = os.getenv("DATA_DIR")
subjects_dir = os.getenv("SUBJECTS_DIR")

if not all([data_dir, subjects_dir]):
    raise ValueError("DATA_DIR or SUBJECTS_DIR not set in .env file or environment.")


BEAMFORMER_OUTPUT_DIR = 'beamformer'

parser = argparse.ArgumentParser(description='EEG Beamformer Source Reconstruction Batch Processing')
parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs to run.')
parser.add_argument('--raw_data_path', type=str, default='raw_eeg/raw_all', help='Path to the raw data directory.')
parser.add_argument('--source_path', type=str, default='source/source_all', help='Path to the source directory.')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output for MNE functions.')
parser.add_argument('--conditions', type=str, nargs='+', default=['Coordination', 'Solo', 'Spontaneous'], help='List of conditions to process. Default: ["Coordination", "Solo", "Spontaneous"]')
args = parser.parse_args()

raw_data_path = os.path.join(data_dir, args.raw_data_path)
source_path = os.path.join(data_dir, args.source_path)
ppts = ['PPT1', 'PPT2']
pair_numbers = list(range(1, 44))
conditions = args.conditions

output_dir = os.path.join(source_path, BEAMFORMER_OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)
omitted_log_path = os.path.join(output_dir, 'omitted_files.log')


# --- Define the processing function with CHECKPOINTING ---
def process_subject(ppt, pair_number, condition):
    """
    This function encapsulates the full beamformer pipeline for one subject file,
    now with checkpointing to skip already completed files.
    """
    s_number = str(100 + pair_number) if ppt == 'PPT1' else str(200 + pair_number)
    file_identifier = f"s_{s_number}_{condition}"
    input_filename = os.path.join(raw_data_path, ppt, f'{file_identifier}.set')

    # 1. Construct the final output filename FIRST.
    output_fname = os.path.join(output_dir, f'{file_identifier}-source-beamformer-epo.fif')

    # 2. Check if the output file already exists.
    if os.path.exists(output_fname):
        # 3. If it exists, print a message and skip this entire function call.
        #    No need for verbose printing here, as we want to quickly get to the work.
        #    You can uncomment the print statement if you want to see what's being skipped.
        print(f"Output file already exists, skipping: {output_fname}")
        return

    # Check if the input file exists (this is for missing raw data)
    if not os.path.exists(input_filename):
        print(f'Skipped missing input file: {input_filename}')
        with open(omitted_log_path, 'a') as log_file:
            log_file.write(f'{file_identifier} (input missing)\n')
        return

    try:
        # If we've reached this point, it means the file needs to be processed.
        print(f'Processing: {file_identifier}')
        log_file = os.path.join(output_dir, f'{file_identifier}.log')

        recon = EEGBeamformerSourceReconstruction(
            source_path=source_path,
            subjects_dir=subjects_dir,
            montage="standard_1005",
            reg=0.05,
            pick_ori='max-power',
            weight_norm='unit-noise-gain',
            verbose=args.verbose,
            log_file=log_file
        )

        recon.load_epochs_eeglab(filename=input_filename)
        recon.compute_forward_solution(fwd_fname='fsaverage_64_fwd.fif', overwrite=False)
        recon.compute_data_covariance()
        recon.make_beamformer_filters()
        label_ts_array, label_names = recon.apply_beamformer_and_extract_rois(
            labels=None, parc='aparc'
        )
        
        # Save the final output using the path we defined earlier.
        recon.save_epochs_array(
            data_3d=label_ts_array, 
            label_names=label_names, 
            out_fname=output_fname
        )

        print(f"Successfully processed and saved: {output_fname}")

    except Exception as e:
        error_message = f'Error processing {file_identifier}: {e}'
        print(error_message)
        with open(omitted_log_path, 'a') as log_file:
            log_file.write(f'{file_identifier} - Error: {error_message}\n')
    
    finally:
        plt.close('all')


if __name__ == "__main__":
    tasks = (
        delayed(process_subject)(ppt, pair_number, condition)
        for ppt in ppts
        for pair_number in pair_numbers
        for condition in conditions
    )

    total_tasks = 2 * len(pair_numbers) * len(conditions)
    print(f"Starting batch processing for {total_tasks} files using {args.n_jobs} jobs...")
    print("Already completed files will be skipped automatically.")
    
    Parallel(n_jobs=args.n_jobs)(tasks)
    
    print("Batch processing complete.")