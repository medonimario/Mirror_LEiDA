import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os

class EEGBeamformerSourceReconstruction:
    """
    A class that performs EEG source reconstruction using an LCMV Beamformer in MNE-Python.

    Workflow:
      1) Load sensor-level epochs from an EEGLAB .set file (or MNE .fif).
      2) Set montage (electrode positions).
      3) Compute or load a pre-computed forward solution.
      4) Compute the data covariance matrix over the time window of interest.
      5) Create the LCMV beamformer filters.
      6) Apply the beamformer to produce ROI-level time series using the 
         Desikan-Killiany or a user-defined atlas.
      7) (Optional) Reshape the ROI time courses back to an epoch structure,
         and/or save outputs and plots.
    """

    def __init__(self,
                 source_path,
                 subjects_dir=None,
                 subject='fsaverage',
                 montage='standard_1005',
                 # Forward settings
                 trans='fsaverage',
                 bem=None,
                 src=None,
                 mindist=5.0,
                 n_jobs=1,
                 # Beamformer settings
                 reg=0.05,
                 pick_ori='max-power',
                 weight_norm='unit-noise-gain',
                 # Logging / output options
                 save_plots=False,
                 verbose=True,
                 log_file=None):
        """
        Parameters
        ----------
        source_path : str
            Directory to save forward solutions, filters, and final source files.
        subjects_dir : str or None
            Path to FreeSurfer subjects directory. If None, MNE will try to
            fetch fsaverage.
        subject : str
            Subject name for forward/inverse computations (e.g., 'fsaverage').
        montage : str
            Name of montage in MNE to set (e.g., 'standard_1005').
        trans : str
            Name of .trans file or 'fsaverage' for template transformation.
        bem : str or None
            Path to .fif BEM solution. If None, assumes fsaverage default.
        src : str or None
            Path to .fif source space. If None, assumes fsaverage default.
        mindist : float
            Minimum distance (in mm) from inner skull for dipoles.
        n_jobs : int
            Number of jobs (threads) for parallel processing.
        reg : float
            Regularization parameter for the covariance matrix.
        pick_ori : str
            Orientation for the beamformer filters: 'max-power', 'normal', 'vector'.
            'max-power' is common for maximizing output signal-to-noise.
        weight_norm : str or None
            Weight normalization method: 'unit-noise-gain', 'nai', or None.
        save_plots : bool
            Whether to save plots generated along the way.
        verbose : bool
            Verbose logging of progress.
        log_file : str or None
            Path to a file for logging messages.
        """

        self.source_path = source_path
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.montage = montage
        self.trans = trans
        self.bem = bem
        self.src = src
        self.mindist = mindist
        self.n_jobs = n_jobs

        # Beamformer parameters
        self.reg = reg
        self.pick_ori = pick_ori
        self.weight_norm = weight_norm

        # Data covariance
        self.data_cov_ = None

        self.save_plots = save_plots
        self.verbose = verbose

        # Will store loaded epochs and any subsequent data
        self.epochs_ = None
        self.fwd_ = None
        self.filters_ = None  # Store beamformer filters here

        self.log_file = log_file

    def _log(self, message):
        """Utility logger method."""
        if self.log_file is not None:
            with open(self.log_file, 'a') as lf:
                lf.write(f"{message}\n")
        if self.verbose:
            print(message)

    def load_epochs_eeglab(self, filename):
        """
        Load preprocessed EEG epochs from an EEGLAB .set file into an MNE Epochs object.
        
        Parameters
        ----------
        filename : str
            Full path to the .set file.
        """
        self._log(f"Loading EEGLAB file: {filename}")
        epochs = mne.io.read_epochs_eeglab(filename, verbose=self.verbose)
        
        self._log(f"Setting montage: {self.montage}")
        montage = mne.channels.make_standard_montage(self.montage)
        epochs.set_montage(montage)

        self._log("Setting average EEG reference.")
        epochs.set_eeg_reference('average', projection=True)

        self._log("Cropping so tmin=0.0")
        epochs.crop(tmin=0.0)
        self._log(f"Epochs tmin: {epochs.tmin}, tmax: {epochs.tmax}")
        
        self._log(f"Loaded epochs with shape {epochs.get_data().shape}.")
        self.epochs_ = epochs

    def load_epochs_fif(self, filename):
        """
        Load preprocessed EEG epochs from an MNE .fif file.
        
        Parameters
        ----------
        filename : str
            Full path to the .fif file containing the epochs.
        """
        self._log(f"Loading MNE .fif epochs file: {filename}")
        epochs = mne.read_epochs(filename, verbose=self.verbose)
        
        self._log(f"Setting montage: {self.montage}")
        montage = mne.channels.make_standard_montage(self.montage)
        epochs.set_montage(montage)

        self._log("Setting average EEG reference.")
        epochs.set_eeg_reference('average', projection=True)
        
        self._log(f"Loaded epochs with shape {epochs.get_data().shape}.")
        self.epochs_ = epochs

    def compute_forward_solution(self, fwd_fname='fsaverage_fwd.fif', overwrite=False):
        """
        Compute or load a forward solution.
        
        Parameters
        ----------
        fwd_fname : str
            Filename (within self.source_path) to save or read the forward solution.
        overwrite : bool
            Whether to overwrite an existing forward solution file.
        """
        if self.fwd_ is not None and not overwrite:
            self._log("Forward solution already computed and overwrite=False. Skipping.")
            return
        
        self.source_path = os.path.expanduser(self.source_path)
        if self.subjects_dir is not None:
            self.subjects_dir = os.path.expanduser(self.subjects_dir)

        os.makedirs(self.source_path, exist_ok=True)
        
        full_fwd_path = os.path.join(self.source_path, fwd_fname)
        if os.path.isfile(full_fwd_path) and not overwrite:
            self._log(f"Loading existing forward solution from {full_fwd_path}")
            self.fwd_ = mne.read_forward_solution(full_fwd_path, verbose=self.verbose)
            return
        
        if not self.subjects_dir:
            self._log("No subjects_dir specified. Attempting to fetch fsaverage via MNE.")
            self.subjects_dir = mne.datasets.fetch_fsaverage(verbose=self.verbose)
            if 'fsaverage' in os.path.basename(self.subjects_dir):
                self.subjects_dir = os.path.dirname(self.subjects_dir)
        
        if self.src is None:
            self.src = os.path.join(self.subjects_dir, self.subject, 'bem', f'{self.subject}-ico-5-src.fif')
        if self.bem is None:
            self.bem = os.path.join(self.subjects_dir, self.subject, 'bem', f'{self.subject}-5120-5120-5120-bem-sol.fif')
        
        self._log("Computing forward solution...")
        fwd = mne.make_forward_solution(
            info=self.epochs_.info, trans=self.trans, src=self.src, bem=self.bem,
            eeg=True, mindist=self.mindist, n_jobs=self.n_jobs, verbose=self.verbose
        )
        self.fwd_ = fwd
        self._log(f"Forward solution computed. Writing to {full_fwd_path}")
        mne.write_forward_solution(full_fwd_path, fwd, overwrite=True)

    def compute_data_covariance(self):
        """
        Compute the data covariance matrix over the entire duration of the epochs.
        This is the required input for the LCMV beamformer.
        """
        if self.epochs_ is None:
            raise RuntimeError("Epochs not loaded. Run load_epochs_...() first.")

        self._log("Computing data covariance over the full epoch duration...")
        self.data_cov_ = mne.compute_covariance(
            self.epochs_,
            tmin=self.epochs_.tmin,
            tmax=self.epochs_.tmax,
            method='empirical', # or 'shrunk' for more robustness
            rank=None, # let MNE estimate rank
            verbose=self.verbose
        )
        self._log("Data covariance computed.")

    def make_beamformer_filters(self):
        """
        Create the LCMV beamformer spatial filters.
        """
        if self.fwd_ is None:
            raise RuntimeError("Forward solution not found. Run compute_forward_solution() first.")
        if self.data_cov_ is None:
            raise RuntimeError("Data covariance not found. Run compute_data_covariance() first.")

        self._log("Creating LCMV beamformer filters...")

        filters = mne.beamformer.make_lcmv(
            self.epochs_.info, # Use the info from our referenced epochs
            self.fwd_,
            self.data_cov_,
            reg=self.reg,
            pick_ori=self.pick_ori,
            weight_norm=self.weight_norm,
            verbose=self.verbose
        )

        self.filters_ = filters
        self._log("Beamformer filters created.")

    def apply_beamformer_and_extract_rois(self,
                                         labels=None,
                                         parc='aparc'):
        """
        Apply the LCMV beamformer filters to the epochs to obtain source time courses,
        then extract the time series for each ROI.
        
        Parameters
        ----------
        labels : list of mne.Label or None
            A list of labels (ROI definitions). If None, reads from annotation
            using `parc` on `self.subject`.
        parc : str
            Parcellation to use, e.g., 'aparc' (Desikan-Killiany).
            
        Returns
        -------
        label_ts_array : ndarray
            The final array of shape (n_epochs, n_labels, n_times) with the source
            time courses in the labeled ROIs.
        label_names : list of str
            The ROI (label) names.
        """
        if self.filters_ is None:
            raise RuntimeError("Beamformer filters not created. Run make_beamformer_filters() first.")
        
        # 1) Load or generate the label set
        if labels is None:
            if not self.subjects_dir:
                raise RuntimeError("subjects_dir not set, cannot read annotations.")
            self._log(f"Reading labels from annotation: subject={self.subject}, parc={parc}")
            labels = mne.read_labels_from_annot(
                subject=self.subject,
                parc=parc,
                subjects_dir=self.subjects_dir
            )
            if labels and 'unknown' in labels[-1].name.lower():
                labels = labels[:-1]
        
        label_names = [lbl.name for lbl in labels]
        
        # 2) Apply beamformer to epochs to get source time courses (STCs)
        self._log("Applying beamformer filters to epochs...")
        stcs = mne.beamformer.apply_lcmv_epochs(
            self.epochs_,
            self.filters_,
            return_generator=False,
            verbose=self.verbose
        )
        
        # 3) Extract ROI time courses from the list of STCs
        self._log(f"Extracting time courses for {len(labels)} ROIs...")
        # This returns a list of 2D arrays (n_labels, n_times), one per epoch.
        label_ts_list = mne.extract_label_time_course(
            stcs,
            labels,
            self.fwd_['src'],
            mode='mean_flip',
            return_generator=False,
            verbose=self.verbose
        )
        
        # Convert the list of 2D arrays into a single 3D NumPy array.
        label_ts_array = np.array(label_ts_list)
        
        self._log(f"ROI-based array shape: {label_ts_array.shape} (n_epochs, n_labels, n_times)")
        
        # Return the new NumPy array
        return label_ts_array, label_names

    def save_epochs_array(self, data_3d, label_names, out_fname='source-epo.fif'):
        """
        Convert the ROI-based data back into an MNE Epochs object and save to disk.
        (Identical to original class)
        """
        # This method is identical to your original one and does not need changes.
        info = mne.create_info(
            ch_names=label_names,
            sfreq=self.epochs_.info['sfreq'],
            ch_types='eeg'
        )
        tmin = self.epochs_.times[0]
        
        label_epochs = mne.EpochsArray(data=data_3d, info=info, tmin=tmin, verbose=self.verbose)
        
        full_out_path = os.path.join(self.source_path, out_fname)
        self._log(f"Saving ROI-level epochs to {full_out_path}")
        label_epochs.save(full_out_path, overwrite=True)

    # --- Plotting and Helper Methods (mostly unchanged) ---
    def quick_plot_data_cov(self):
        """(Optional) Quick plot of the data covariance matrix."""
        if self.data_cov_ is None:
            raise RuntimeError("No data covariance computed.")
        self._log("Plotting data covariance matrix...")
        mne.viz.plot_cov(self.data_cov_, info=self.epochs_.info, show=True)

    def quick_plot_sensors(self):
        """(Optional) Quick plot of sensor positions in 3D."""
        if self.epochs_ is None: raise RuntimeError("No epochs loaded.")
        self._log("Plotting sensor layout in 3D...")
        self.epochs_.plot_sensors(kind='3d')

    def quick_plot_alignment(self):
        """(Optional) Quick plot for checking alignment of EEG sensors and MRI."""
        if self.fwd_ is None: raise RuntimeError("Need a forward solution to plot alignment.")
        self._log("Plotting alignment of sensors, src, and BEM...")
        mne.viz.plot_alignment(self.epochs_.info, trans=self.trans, subject=self.subject,
                              bem=self.bem, src=self.fwd_['src'], meg=False,
                              eeg=['original', 'projected'], subjects_dir=self.subjects_dir,
                              show_axes=True, dig='fiducials')
                              
    def plot_sensor_vs_roi_epochs(self, sensor_epochs, roi_epochs, sensor_name,
                                  roi_name, invert_roi=False, epoch_idx=None):
        """
        Plot a sensor-level channel vs. an ROI-level channel from MNE Epochs objects.
        """
        if sensor_name not in sensor_epochs.ch_names:
            raise ValueError(f"Sensor channel '{sensor_name}' not found.")
        if roi_name not in roi_epochs.ch_names:
            raise ValueError(f"ROI channel '{roi_name}' not found.")

        sensor_idx = sensor_epochs.ch_names.index(sensor_name)
        roi_idx = roi_epochs.ch_names.index(roi_name)

        if epoch_idx is not None:
            sensor_signal = sensor_epochs.get_data(picks=[sensor_idx])[epoch_idx, 0, :]
            roi_signal = roi_epochs.get_data(picks=[roi_idx])[epoch_idx, 0, :]
            time = sensor_epochs.times
            title_suffix = f" (Epoch {epoch_idx})"
        else:
            sensor_data = sensor_epochs.get_data(picks=[sensor_idx])[:, 0, :]
            roi_data = roi_epochs.get_data(picks=[roi_idx])[:, 0, :]
            sensor_signal = sensor_data.reshape(-1)
            roi_signal = roi_data.reshape(-1)
            epoch_time = sensor_epochs.times
            n_epochs = len(sensor_epochs)
            epoch_duration = epoch_time[-1] - epoch_time[0]
            time = np.array([t + i * epoch_duration for i in range(n_epochs) for t in epoch_time])
            title_suffix = f" (All {n_epochs} epochs concatenated)"

        if invert_roi:
            roi_signal *= -1
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time, sensor_signal)
        axes[0].set_ylabel('Sensor Amplitude')
        axes[0].set_title(f"Sensor-level: {sensor_name}{title_suffix}")
        axes[1].plot(time, roi_signal)
        axes[1].set_ylabel('ROI Amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f"Source-level: {roi_name}{title_suffix}")
        plt.tight_layout()
        plt.show()

        sensor_norm = zscore(sensor_signal)
        roi_norm = zscore(roi_signal)
        plt.figure(figsize=(12, 4))
        plt.plot(time, sensor_norm, label=f"{sensor_name} (Sensor)", linewidth=1.5)
        plt.plot(time, roi_norm, label=f"{roi_name} (ROI)", linewidth=1.5)
        plt.title(f"Z-scored signals{title_suffix}")
        plt.xlabel("Time (s)")
        plt.ylabel("Z-scored amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    mne.viz.set_3d_backend("pyvistaqt")
    # Example usage for subject 101, condition Coordination with BEAMFORMING
    recon = EEGBeamformerSourceReconstruction(
        source_path="data/source_reconstruction_beamformer/",
        subjects_dir="~/mne_data/MNE-fsaverage-data/",
        montage="standard_1005",
        # --- Beamformer-specific parameters ---
        reg=0.05,                   # Regularization for covariance. 5% is a good starting point.
        pick_ori='max-power',       # Max-power orientation is great for time-series analysis.
        weight_norm='unit-noise-gain', # Preserves the units of the original data.
        verbose=True,
        log_file="beamformer_reconstruction.txt"
    )

    # Create output directory if it doesn't exist
    os.makedirs(recon.source_path, exist_ok=True)
    
    # 1. Load EEGLAB epochs
    recon.load_epochs_eeglab(filename="data/raw/PPT1/s_101_Coordination.set")

    # 2. Compute forward solution
    recon.compute_forward_solution(fwd_fname='fsaverage_64_fwd.fif', overwrite=False)

    # 3. Compute DATA covariance (this is the key change)
    recon.compute_data_covariance()

    # 4. Make beamformer filters
    recon.make_beamformer_filters()

    # 5. Apply beamformer and extract ROIs
    label_ts_array, label_names = recon.apply_beamformer_and_extract_rois(
        labels=None,   # Will read from 'aparc'
        parc='aparc'   # Desikan-Killiany
    )

    # 6. Save the epochs array
    recon.save_epochs_array(
        data_3d=label_ts_array,
        label_names=label_names,
        out_fname='s_101_Coordination-source-beamformer-epo.fif'
    )

    # --- Optional Visualization ---
    # Apply the beamformer to the evoked (averaged) data to get a clean power map
    evoked = recon.epochs_.average()
    stc = mne.beamformer.apply_lcmv(evoked, recon.filters_)

    # Plot the source power on the brain
    # Use a threshold to only show the strongest sources.
    brain = stc.plot(
        surface='inflated',
        hemi='both',
        subjects_dir=recon.subjects_dir,
        subject='fsaverage',
        time_label='Source Power',
        initial_time=0.1, # Pick a time point to view
        
        clim=dict(kind='percent', lims=[95, 97.5, 100]) # Show top 5% of activity
    )
    
    # Plot data covariance
    recon.quick_plot_data_cov()
    # Plot sensor layout
    recon.quick_plot_sensors()
    # Plot alignment
    recon.quick_plot_alignment()

    # Compare sensor and source waveforms
    roi_epochs_path = os.path.join(recon.source_path, 's_101_Coordination-source-beamformer-epo.fif')
    roi_epochs = mne.read_epochs(roi_epochs_path)

    recon.plot_sensor_vs_roi_epochs(
        sensor_epochs=recon.epochs_,
        roi_epochs=roi_epochs,
        sensor_name='Pz',                      # Example sensor
        roi_name='precuneus-rh',               # Example ROI that is spatially close to Pz
        invert_roi=False,
        epoch_idx=0) # Plot the first epoch
    
    