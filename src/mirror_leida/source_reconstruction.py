import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore

class EEGSourceReconstruction:
    """
    A class that performs EEG source reconstruction using MNE-Python.
    
    Workflow:
      1) Load sensor-level epochs from an EEGLAB .set file (or MNE .fif).
      2) Set montage (electrode positions).
      3) (Optional) Compute or load a pre-computed forward solution.
      4) Create or load noise covariance (ad-hoc or data-driven).
      5) Make the inverse operator based on user settings (e.g., MNE, loose, depth).
      6) Apply inverse solution to produce ROI-level time series using the 
         Desikan-Killiany or user-defined atlas.
      7) (Optional) Reshape the ROI time courses back to an epoch structure, 
         and/or save outputs and plots.
    """
    
    def __init__(self,
                 raw_data_path,
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
                 # Inverse settings
                 method='MNE',
                 loose=1.0,
                 depth=0.8,
                 snr=3.0,
                 # Noise covariance
                 noise_cov_type='ad-hoc',  
                 # Logging / output options
                 save_plots=False,
                 verbose=True):
        """
        Parameters
        ----------
        raw_data_path : str
            Directory where raw EEG data (e.g., .set files) is located.
        source_path : str
            Directory where to save forward solutions, inverse operators, 
            and final source-reconstructed files.
        subjects_dir : str or None
            Path to FreeSurfer subjects directory. If None, MNE will try to 
            fetch fsaverage or use what is set in the environment.
        subject : str
            Subject name for forward/inverse computations (e.g., 'fsaverage').
        montage : str
            Name of montage in MNE to set (e.g., 'standard_1005' or 'biosemi64').
        trans : str
            Name of .trans file or 'fsaverage' when using MNE's built-in 
            template transformation.
        bem : str or None
            Path to .fif BEM solution. If None, will assume fsaverage default 
            or let user specify at runtime.
        src : str or None
            Path to .fif source space. If None, will assume fsaverage default.
        mindist : float
            Minimum distance (in mm) from inner skull for dipoles in forward model.
        n_jobs : int
            Number of jobs (threads) for parallel processing.
        method : str
            Inverse method: 'MNE', 'dSPM', or 'sLORETA'.
        loose : float
            Loose parameter for the inverse operator (0 <= loose <= 1).
        depth : float
            Depth parameter for inverse operator.
        snr : float
            Signal-to-noise ratio for regularization. lambda2 = 1.0/snr^2.
        noise_cov_type : str
            Type of noise covariance to use: 'ad-hoc' or 'precomputed'.
        save_plots : bool
            Whether or not to save plots generated along the way.
        verbose : bool
            Verbose logging of progress.
        """
        
        self.raw_data_path = raw_data_path
        self.source_path = source_path
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.montage = montage
        self.trans = trans
        self.bem = bem
        self.src = src
        self.mindist = mindist
        self.n_jobs = n_jobs
        
        # Inverse parameters
        self.method = method
        self.loose = loose
        self.depth = depth
        self.snr = snr
        self.lambda2 = 1.0 / (snr ** 2)
        
        # Noise covariance
        self.noise_cov_type = noise_cov_type
        self.noise_cov_ = None
        
        self.save_plots = save_plots
        self.verbose = verbose
        
        # Will store loaded epochs and any subsequent data
        self.epochs_ = None
        self.fwd_ = None
        self.inverse_operator_ = None
    
    def _log(self, message):
        """Utility logger method."""
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
        
        self._log(f"Loaded epochs with shape {epochs.get_data().shape}.")
        self.epochs_ = epochs

    def compute_forward_solution(self, fwd_fname='fsaverage_fwd.fif', overwrite=False):
        """
        Compute or load a forward solution for fsaverage or a specified subject.
        
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
        
        # Attempt to load from file if exists and not overwriting
        full_fwd_path = os.path.join(self.source_path, fwd_fname)
        if os.path.isfile(full_fwd_path) and not overwrite:
            self._log(f"Loading existing forward solution from {full_fwd_path}")
            self.fwd_ = mne.read_forward_solution(full_fwd_path, verbose=self.verbose)
            return
        
        if not self.subjects_dir:
            # Attempt to fetch fsaverage or rely on default
            self._log("No subjects_dir specified. Attempting to fetch fsaverage via MNE.")
            self.subjects_dir = mne.datasets.fetch_fsaverage(verbose=self.verbose)
            if 'fsaverage' in os.path.basename(self.subjects_dir):
                self.subjects_dir = os.path.dirname(self.subjects_dir)
        
        if self.src is None:
            self.src = os.path.join(self.subjects_dir,
                                    self.subject,
                                    'bem',
                                    f'{self.subject}-ico-5-src.fif')
        if self.bem is None:
            self.bem = os.path.join(self.subjects_dir,
                                    self.subject,
                                    'bem',
                                    f'{self.subject}-5120-5120-5120-bem-sol.fif')
        
        self._log("Computing forward solution...")
        fwd = mne.make_forward_solution(
            info=self.epochs_.info,
            trans=self.trans,
            src=self.src,
            bem=self.bem,
            eeg=True,
            mindist=self.mindist,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        self.fwd_ = fwd
        self._log(f"Forward solution computed. Writing to {full_fwd_path}")
        mne.write_forward_solution(full_fwd_path, fwd, overwrite=True)

    def compute_noise_covariance(self, baseline_times=None, baseline_raw=None):
        """
        Compute noise covariance matrix. Supports either an 'ad-hoc' method 
        or a data-driven approach using MNE's compute_covariance function.

        Parameters
        ----------
        baseline_times : tuple of float, optional
            Start and end times for baseline period (in seconds).
            If None, `baseline_raw` must be provided.
        baseline_raw : str, optional
            Path to a raw EEG file for computing covariance.
            If None, `baseline_times` must be provided.

        Raises
        ------
        ValueError
            If neither `baseline_times` nor `baseline_raw` is provided.
        """

        if self.noise_cov_type == 'ad-hoc':
            self._log("Using ad-hoc noise covariance.")
            self.noise_cov_ = mne.make_ad_hoc_cov(self.epochs_.info)
            return

        if baseline_times is None and baseline_raw is None:
            raise ValueError("Either baseline_times or baseline_raw must be specified.")

        if baseline_raw:
            self._log(f"Loading baseline from file: {baseline_raw}")
            raw_baseline = mne.io.read_raw_eeglab(baseline_raw, verbose=self.verbose)

            # Set baseline times to the full duration of the raw file if not provided
            tmin, tmax = raw_baseline.times[0], raw_baseline.times[-1] if baseline_times is None else baseline_times

            self.noise_cov_ = mne.compute_covariance(
                raw_baseline, tmin=tmin, tmax=tmax, method='empirical', verbose=self.verbose
            )

        elif isinstance(baseline_times, tuple):
            self._log(f"Computing noise covariance from baseline: {baseline_times}")
            self.noise_cov_ = mne.compute_covariance(
                self.epochs_, tmin=baseline_times[0], tmax=baseline_times[1], method='empirical', verbose=self.verbose
            )

    def make_inverse_operator(self):
        """
        Create or load the inverse operator using the user-specified parameters.
        
        Parameters
        ----------
        inv_fname : str
            Filename (within self.source_path) for storing/loading the inverse operator.
        overwrite : bool
            Whether to overwrite if an existing file is found.
        """
        if self.fwd_ is None:
            raise RuntimeError("Forward solution not found. Please run compute_forward_solution() first.")
        if self.noise_cov_ is None:
            raise RuntimeError("Noise covariance not found. Please run compute_noise_covariance() first.")

        self._log("Creating inverse operator...")
        inv_op = mne.minimum_norm.make_inverse_operator(
            info=self.epochs_.info,
            forward=self.fwd_,
            noise_cov=self.noise_cov_,
            loose=self.loose,
            depth=self.depth,
            verbose=self.verbose
        )
        self.inverse_operator_ = inv_op
        self._log("Inverse operator created.")

    def apply_inverse_and_extract_rois(self, 
                                       labels=None, 
                                       parc='aparc', 
                                       pick_ori='vector',
                                       pca_directions='pca'):
        """
        Apply the inverse operator to the entire concatenated data 
        to obtain label time series for each ROI.
        
        Parameters
        ----------
        labels : list of mne.Label or None
            A list of labels (ROI definitions). If None, will read from annotation 
            using the provided `parc` on self.subject in self.subjects_dir.
        parc : str
            Which parcellation to use, e.g. 'aparc' (Desikan-Killiany), 'aparc.a2009s', etc.
        pick_ori : str
            'vector' for free orientation, 'normal' for normal orientation, or None 
            for loose orientation. Must be consistent with how inverse was made.
        pca_directions : str
            If 'pca', the method uses principal components to collapse orientation 
            dimension. 'mean' would just average across orientation.
            
        Returns
        -------
        label_ts_reshaped : ndarray
            The final array of shape (n_epochs, n_labels, n_times) with the source 
            time courses in the labeled ROIs.
        label_names : list of str
            The ROI (label) names.
        """
        
        if self.inverse_operator_ is None:
            raise RuntimeError("Inverse operator not set. Run make_inverse_operator() first.")
        
        # 1) Concatenate epochs into one raw
        data_3d = self.epochs_.get_data(picks='eeg')  # shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = data_3d.shape
        data_2d = data_3d.transpose(1, 0, 2).reshape(n_channels, -1)  # -> [n_channels, n_epochs*n_times]
        
        info_eeg = self.epochs_.copy().pick_types(eeg=True).info
        # Create an MNE RawArray from the concatenated data
        raw = mne.io.RawArray(data_2d, info_eeg, verbose=self.verbose)
        raw._filenames = [""]  # to avoid warnings about missing filename
        raw.set_eeg_reference(projection=True)
        
        # 2) (Optional) load or generate the label set
        if labels is None:
            if not self.subjects_dir:
                raise RuntimeError("subjects_dir not set, cannot read annotations. "
                                   "Provide a custom labels list or set subjects_dir.")
            self._log(f"Reading labels from annotation: subject={self.subject}, parc={parc}")
            labels = mne.read_labels_from_annot(subject=self.subject,
                                                parc=parc,
                                                subjects_dir=self.subjects_dir)
            # remove unknown label if desired
            if labels and 'unknown' in labels[-1].name.lower():
                labels = labels[:-1]
        
        label_names = [lbl.name for lbl in labels]
        
        # 3) Prepare to apply inverse
        inv_op = self.inverse_operator_
        n_labels = len(labels)
        label_ts = np.zeros((n_labels, data_2d.shape[1]))  # shape [n_labels, total_time_points]
        
        self._log(f"Applying inverse to {n_labels} labels. pick_ori={pick_ori}")
        
        # 4) For each label, apply inverse restricted to that label
        # This is memory-efficient for large data.
        for li, label in enumerate(labels):
            # Solve inverse only for this label
            stc = mne.minimum_norm.apply_inverse_raw(
                raw,
                inv_op,
                lambda2=self.lambda2,
                method=self.method,
                pick_ori=pick_ori,
                label=label,
                verbose=False
            )
            # Collapse orientation (3 components) if vector
            if pick_ori == 'vector':
                stc_pca, _ = stc.project(directions=pca_directions, src=inv_op['src'])
            else:
                stc_pca = stc  # no extra step
            
            # Extract mean flip time course
            roi_data = mne.extract_label_time_course(
                stc_pca, [label], inv_op['src'],
                mode='mean_flip', return_generator=False, verbose=False
            )
            label_ts[li, :] = roi_data[0, :]
            
        # 5) Reshape to [n_epochs, n_labels, n_times]
        label_ts_reshaped = label_ts.reshape(n_labels, n_epochs, n_times).transpose(1, 0, 2)
        
        self._log(f"ROI-based array shape: {label_ts_reshaped.shape} "
                  f"(n_epochs, n_labels, n_times)")
        
        return label_ts_reshaped, label_names

    def save_epochs_array(self, data_3d, label_names, out_fname='source-epo.fif'):
        """
        Convert the ROI-based data back into MNE Epochs object, 
        and save to disk as a .fif file.
        
        Parameters
        ----------
        data_3d : ndarray of shape (n_epochs, n_labels, n_times)
            The ROI-based time courses.
        label_names : list of str
            Names for each ROI channel.
        out_fname : str
            Path (within self.source_path) of the .fif file to write.
        """
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
    
    def quick_plot_sensors(self):
        """(Optional) Quick plot of sensor positions in 3D."""
        if self.epochs_ is None:
            raise RuntimeError("No epochs loaded.")
        self._log("Plotting sensor layout in 3D...")
        self.epochs_.plot_sensors(kind='3d')

    def quick_plot_alignment(self):
        """(Optional) Quick plot for checking alignment of EEG sensors and MRI."""
        if self.fwd_ is None:
            raise RuntimeError("Need a forward solution to plot alignment.")
        self._log("Plotting alignment of sensors, src, and BEM...")
        mne.viz.plot_alignment(self.epochs_.info,
                              trans=self.trans,
                              subject=self.subject,
                              bem=self.bem,
                              src=self.fwd_['src'],
                              meg=False,
                              eeg=['original', 'projected'],
                              subjects_dir=self.subjects_dir,
                              show_axes=True,
                              dig='fiducials')

    def plot_sensor_vs_roi(self, raw, roi_raw, sensor_name, roi_name, invert_roi=False):
        """
        Example method to plot a sensor-level channel vs. a ROI-level time course.
        
        Parameters
        ----------
        raw : mne.io.RawArray
            The concatenated sensor-level data.
        roi_raw : mne.io.RawArray
            The concatenated ROI-level data.
        sensor_name : str
            Name of the sensor in raw.ch_names.
        roi_name : str
            Name of the ROI channel in roi_raw.ch_names.
        invert_roi : bool
            Whether to multiply the ROI signal by -1 before plotting 
            (sometimes helpful to visually match phases).
        """
        
        if sensor_name not in raw.ch_names:
            raise ValueError(f"Sensor channel {sensor_name} not in raw.")
        if roi_name not in roi_raw.ch_names:
            raise ValueError(f"ROI channel {roi_name} not in roi_raw.")
        
        sensor_idx = raw.ch_names.index(sensor_name)
        roi_idx = roi_raw.ch_names.index(roi_name)
        
        sensor_signal = raw.get_data(picks=[sensor_idx])[0]
        roi_signal = roi_raw.get_data(picks=[roi_idx])[0]
        if invert_roi:
            roi_signal *= -1
        
        sfreq = raw.info['sfreq']
        n_times = sensor_signal.shape[0]
        time = np.arange(n_times) / sfreq
        
        # Plot side by side
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(time, sensor_signal, color='blue')
        axes[0].set_ylabel('Sensor amplitude (µV)')
        axes[0].set_title(f'Sensor-level signal: {sensor_name}')
        
        axes[1].plot(time, roi_signal, color='green')
        axes[1].set_ylabel('Source amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f'Source-level signal: {roi_name}')
        
        plt.tight_layout()
        plt.show()
        
        # Also show normalized (z-score) signals for comparison
        sensor_norm = zscore(sensor_signal)
        roi_norm = zscore(roi_signal)
        
        plt.figure(figsize=(12,5))
        plt.plot(time, sensor_norm, label=f'{sensor_name} (Sensor)', linewidth=2)
        plt.plot(time, roi_norm, label=f'{roi_name} (ROI)', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Z-scored amplitude')
        plt.title('Normalized ROI vs. Sensor waveforms')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage for subject 101, condition Coordination
    recon = EEGSourceReconstruction(
        raw_data_path="../data/raw/",
        source_path="../data/source_reconstruction/",
        subjects_dir="~/mne_data/freesurfer/subjects",  # Or wherever your FS data is
        montage="standard_1005",
        method="MNE",
        loose=1.0,
        depth=0.8,
        snr=3.0,
        noise_cov_type='ad-hoc',
        verbose=True
    )

    # Load EEGLAB epochs
    recon.load_epochs_eeglab(filename="../data/raw/PPT1/s_101_Coordination.set")

    # Compute forward solution
    recon.compute_forward_solution(fwd_fname='fsaverage_64_fwd.fif', overwrite=False)

    # Compute noise covariance
    recon.compute_noise_covariance(baseline_times=None, baseline_raw=None)

    # Make inverse operator
    recon.make_inverse_operator()

    # Apply inverse and extract ROIs
    label_ts_reshaped, label_names = recon.apply_inverse_and_extract_rois(
        labels=None,   # Will read from 'aparc'
        parc='aparc',  # Desikan-Killiany
        pick_ori='vector' 
    )

    # Save the epochs array
    recon.save_epochs_array(
        data_3d=label_ts_reshaped,
        label_names=label_names,
        out_fname='s_101_Coordination-source-epo.fif'
    )

    # Quick plot sensor layout
    recon.quick_plot_sensors()
    # Quick plot alignment
    recon.quick_plot_alignment()
    # Example of plotting sensor vs. ROI
    recon.plot_sensor_vs_roi(
        raw=recon.epochs_.get_data(),
        roi_raw=label_ts_reshaped,
        sensor_name='P1',
        roi_name='superiorparietal-lh'
    )








