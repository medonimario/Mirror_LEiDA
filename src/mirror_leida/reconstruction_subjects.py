import mne
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import argparse

# Make sure your modified EEGSourceReconstruction includes:
#   - self.log_file in __init__
#   - _log() method that appends messages to that file
from source_reconstruction import EEGSourceReconstruction

# Load the .env file
load_dotenv()

data_dir = os.getenv("DATA_DIR")
print(f"Data directory is: {data_dir}")
subjects_dir = os.getenv("SUBJECTS_DIR")
print(f"Subjects directory is: {subjects_dir}")

raw_data_path = os.path.join(data_dir, 'raw')
source_path = os.path.join(data_dir, 'source')

parser = argparse.ArgumentParser(description='EEG Source Reconstruction')

parser.add_argument('--method', type=str, default='dSPM', help='Reconstruction method to use')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')

args = parser.parse_args()

ppts = ['PPT1', 'PPT2']
pair_numbers = [i for i in range(1, 44)] 
conditions = ['Coordination', 'Solo', 'Spontaneous']

for ppt in ppts:
    for pair_number in pair_numbers:
        s_number = str(100 + int(pair_number)) if ppt == 'PPT1' else str(200 + int(pair_number))
        for condition in conditions:
            print(50 * '#')
            print(' ')
            print(f'Processing {ppt} {s_number} {condition}')
            print(' ')
            print(50 * '#')
            
            # Decide where to store the logs (in the same folder as the output data)
            output_dir = os.path.join(source_path, args.method)
            os.makedirs(output_dir, exist_ok=True)
            log_fname = f's_{s_number}_{condition}.log'
            log_file = os.path.join(output_dir, log_fname)

            # Instantiate EEGSourceReconstruction with log_file
            recon = EEGSourceReconstruction(
                source_path=source_path,
                subjects_dir=subjects_dir,
                montage="standard_1005",
                method=args.method,
                loose=1.0,
                depth=0.8,
                snr=3.0,
                verbose=args.verbose,
                log_file=log_file 
            )
        
            # Load EEGLAB epochs
            filename = os.path.join(raw_data_path, ppt, f's_{s_number}_{condition}.set')
            recon.load_epochs_eeglab(filename=filename)

            # Compute forward solution
            recon.compute_forward_solution(fwd_fname='fsaverage_64_fwd.fif', overwrite=False)

            # Compute noise covariance
            recon.compute_noise_covariance(baseline_times=None, baseline_epochs=None)

            # Make inverse operator
            recon.make_inverse_operator()

            # Apply inverse and extract ROIs
            label_ts_reshaped, label_names = recon.apply_inverse_and_extract_rois(
                labels=None,   # Will read from 'aparc'
                parc='aparc',  # Desikan-Killiany
                pick_ori='vector' 
            )

            out_fname = os.path.join(
                output_dir,
                f's_{s_number}_{condition}-source.fif'
            )
            # Save the epochs array
            recon.save_epochs_array(
                data_3d=label_ts_reshaped,
                label_names=label_names,
                out_fname=out_fname
            )
