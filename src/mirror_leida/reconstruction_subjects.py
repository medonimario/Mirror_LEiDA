import os
import argparse
from dotenv import load_dotenv
from joblib import Parallel, delayed

from source_reconstruction import EEGSourceReconstruction

# Load .env file
load_dotenv()
data_dir = os.getenv("DATA_DIR")
subjects_dir = os.getenv("SUBJECTS_DIR")

raw_data_path = os.path.join(data_dir, 'raw')
source_path = os.path.join(data_dir, 'source')

parser = argparse.ArgumentParser(description='EEG Source Reconstruction')
parser.add_argument('--method', type=str, default='dSPM', help='Reconstruction method to use')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')

args = parser.parse_args()

ppts = ['PPT1', 'PPT2']
pair_numbers = list(range(1, 44))
conditions = ['Coordination', 'Solo', 'Spontaneous']

# Function to encapsulate the processing per subject-condition combination
def process_subject(ppt, pair_number, condition):
    s_number = str(100 + pair_number) if ppt == 'PPT1' else str(200 + pair_number)
    
    print(f'Processing {ppt} {s_number} {condition}')

    output_dir = os.path.join(source_path, args.method)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f's_{s_number}_{condition}.log')

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

    filename = os.path.join(raw_data_path, ppt, f's_{s_number}_{condition}.set')
    recon.load_epochs_eeglab(filename=filename)
    recon.compute_forward_solution(fwd_fname='fsaverage_64_fwd.fif', overwrite=False)
    recon.compute_noise_covariance(baseline_times=None, baseline_epochs=None)
    recon.make_inverse_operator()
    label_ts_reshaped, label_names = recon.apply_inverse_and_extract_rois(
        labels=None, parc='aparc', pick_ori='vector'
    )
    
    out_fname = os.path.join(output_dir, f's_{s_number}_{condition}-source.fif')
    recon.save_epochs_array(data_3d=label_ts_reshaped, label_names=label_names, out_fname=out_fname)

# Parallel processing call
Parallel(n_jobs=-1)(
    delayed(process_subject)(ppt, pair_number, condition)
    for ppt in ppts
    for pair_number in pair_numbers
    for condition in conditions
)
