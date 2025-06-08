import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
                 # Logging / output options
                 save_plots=False,
                 verbose=True,
                 log_file=None):
        """
        Parameters
        ----------
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
        save_plots : bool
            Whether or not to save plots generated along the way.
        verbose : bool
            Verbose logging of progress.
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
        
        # Inverse parameters
        self.method = method
        self.loose = loose
        self.depth = depth
        self.snr = snr
        self.lambda2 = 1.0 / (snr ** 2)
        
        # Noise covariance
        self.noise_cov_ = None
        
        self.save_plots = save_plots
        self.verbose = verbose
        
        # Will store loaded epochs and any subsequent data
        self.epochs_ = None
        self.fwd_ = None
        self.inverse_operator_ = None

        self.log_file = log_file
    
    def _log(self, message):
        """Utility logger method."""
        # Always write to file if a log_file is provided
        if self.log_file is not None:
            with open(self.log_file, 'a') as lf:
                lf.write(f"{message}\n")
        # Also print to console if verbose is True
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

        self._log(f"Cropping so tmin=0.0")
        epochs.crop(tmin=0.0)
        self._log(f"Empochs tmin: {epochs.tmin}, tmax: {epochs.tmax}")
        
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
        
        # Resolve source_path and subjects_dir if they contain "~"
        self.source_path = os.path.expanduser(self.source_path)
        if self.subjects_dir is not None:
            self.subjects_dir = os.path.expanduser(self.subjects_dir)

        # Make sure the output directory exists
        os.makedirs(self.source_path, exist_ok=True)
        
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

    def compute_noise_covariance(self, baseline_times=None, baseline_epochs=None):
        """
        Compute noise covariance matrix. Supports either an 'ad-hoc' method 
        or a data-driven approach using MNE's compute_covariance function.

        Parameters
        ----------
        baseline_times : tuple of float, optional
            Start and end times for baseline period (in seconds).
            If None, `baseline_epochs` must be provided.
        baseline_epochs : str, optional
            Path to a epochs EEG file for computing covariance.
            If None, `baseline_times` must be provided.

        Raises
        ------
        ValueError
            If neither `baseline_times` nor `baseline_epochs` is provided.
        """

        if baseline_times is None and baseline_epochs is None:
            self._log("Using ad-hoc noise covariance.")
            self.noise_cov_ = mne.make_ad_hoc_cov(self.epochs_.info)
            return 

        if baseline_epochs:
            self._log(f"Loading baseline from file: {baseline_epochs}")
            baseline_epochs = mne.io.read_epochs_eeglab(baseline_epochs, verbose=self.verbose)
            baseline_epochs.crop(tmax=-0.5)
            self._log(f"Loaded baseline epochs with shape {baseline_epochs.get_data().shape}.")

            # Set baseline times to the full duration of baseline_epochs
            tmin, tmax = baseline_epochs.times[0], baseline_epochs.times[-1] if baseline_times is None else baseline_times

            self.noise_cov_ = mne.compute_covariance(
                baseline_epochs, tmin=tmin, tmax=tmax, method='empirical', verbose=self.verbose
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
            self._log(f"Label {li+1}/{n_labels} processed: {label.name}")
            
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

    def quick_plot_noise_cov(self):
        """(Optional) Quick plot of noise covariance matrix."""
        if self.noise_cov_ is None:
            raise RuntimeError("No noise covariance computed.")
        self._log("Plotting noise covariance matrix...")
        mne.viz.plot_cov(self.noise_cov_, info=self.epochs_.info, show=True)

    
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

    def plot_sensor_vs_roi_epochs(self,
                                sensor_epochs,
                                roi_epochs,
                                sensor_name,
                                roi_name,
                                invert_roi=False,
                                epoch_idx=None):  # Made optional
        """
        Plot a sensor-level channel vs. an ROI-level channel from MNE Epochs objects.
        By default, concatenates all epochs to show the entire signal.

        Parameters
        ----------
        sensor_epochs : mne.Epochs
            MNE Epochs containing sensor-level EEG data.
        roi_epochs : mne.Epochs
            MNE Epochs containing ROI-level data (e.g., from source reconstruction).
            Each ROI should appear as a 'channel' in roi_epochs.ch_names.
        sensor_name : str
            Name of the sensor channel in sensor_epochs.ch_names.
        roi_name : str
            Name of the ROI channel in roi_epochs.ch_names.
        invert_roi : bool
            Whether to multiply the ROI signal by -1 before plotting.
        epoch_idx : int, optional
            If provided, plot only this specific epoch. If None (default),
            concatenate all epochs to show the entire signal.
        """

        # Sanity checks
        if sensor_name not in sensor_epochs.ch_names:
            raise ValueError(f"Sensor channel '{sensor_name}' not found in sensor_epochs.")
        if roi_name not in roi_epochs.ch_names:
            raise ValueError(f"ROI channel '{roi_name}' not found in roi_epochs.")

        # Get the channel indices
        sensor_idx = sensor_epochs.ch_names.index(sensor_name)
        roi_idx = roi_epochs.ch_names.index(roi_name)

        # Extract the data
        if epoch_idx is not None:
            # Extract single epoch if specified
            sensor_signal = sensor_epochs.get_data(picks=[sensor_idx])[epoch_idx, 0, :]
            roi_signal = roi_epochs.get_data(picks=[roi_idx])[epoch_idx, 0, :]
            time = sensor_epochs.times
            title_suffix = f" (Epoch {epoch_idx})"
        else:
            # Concatenate all epochs
            sensor_data = sensor_epochs.get_data(picks=[sensor_idx])[:, 0, :]  # [n_epochs, n_times]
            roi_data = roi_epochs.get_data(picks=[roi_idx])[:, 0, :]  # [n_epochs, n_times]
            
            # Flatten across epochs
            sensor_signal = sensor_data.reshape(-1)
            roi_signal = roi_data.reshape(-1)
            
            # Create time vector that spans all epochs
            epoch_time = sensor_epochs.times
            n_epochs = len(sensor_epochs)
            epoch_duration = epoch_time[-1] - epoch_time[0]
            
            # Create concatenated time array
            time = np.zeros(len(sensor_signal))
            for i in range(n_epochs):
                start_idx = i * len(epoch_time)
                end_idx = (i + 1) * len(epoch_time)
                time[start_idx:end_idx] = epoch_time + i * epoch_duration
            
            title_suffix = f" (All {n_epochs} epochs concatenated)"

        if invert_roi:
            roi_signal *= -1

        # Basic plot of the two time series (sensor vs. ROI)
        
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

        # Also plot normalized waveforms (z-scores) for easier comparison
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

    # ---------------------------------------------------------------------
    def assess_data_fit(self, *, pick_ori='vector', metric='r2'):
        """
        Quantify how well the inverse + forward model reproduce the sensor data.

        This version forces 'MNE' at apply time so that `stc.data` are in Am,
        which is required for a valid forward projection.

        Parameters
        ----------
        pick_ori : {'vector', 'normal', None}
            Orientation used when applying the inverse. Must match how the inverse
            operator was created (i.e. if inverse was built loose=1.0, then pick_ori='vector').
        metric : {'r2', 'corr'}
            • 'r2'   → coefficient of determination (explained variance)
            • 'corr' → Pearson correlation

        Returns
        -------
        fit_per_channel : dict
            Keys are channel names; values are the averaged fit metric
            (across all epochs) for that channel.
        """

        # 1) Sanity checks
        if self.inverse_operator_ is None or self.fwd_ is None or self.epochs_ is None:
            raise RuntimeError("Need self.epochs_, self.fwd_, and self.inverse_operator_ first.")

        # 2) Copy epochs, apply average-reference projection
        epochs_ref = self.epochs_.copy()
        epochs_ref.set_eeg_reference('average', projection=True)
        epochs_ref.apply_proj()

        # Extract data array: shape = (n_epochs, n_ch, n_times)
        data_array = epochs_ref.get_data(picks='all')
        n_epochs, n_ch, n_times = data_array.shape

        # 3) Force MNE (raw current) at apply time, even if inverse was built as dSPM
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs_ref,
            self.inverse_operator_,
            lambda2=self.lambda2,
            method='MNE',            # <-- force raw‐current MNE output
            pick_ori=pick_ori,
            verbose=False
        )

        # 4) Loop over epochs: forward‐project and compare
        fit_accumulator = np.zeros(n_ch)
        for idx, stc in enumerate(stcs):
            # (a) Forward‐project. Now stc.data is in Am (because method='MNE').
            pred_evoked = mne.apply_forward(
                self.fwd_, stc, info=epochs_ref.info, verbose=False
            )
            pred = pred_evoked.data  # shape = (n_ch, n_times)

            # (b) Extract the projected original data
            orig = data_array[idx]   # shape = (n_ch, n_times)

            # (c) Compute metric per channel
            if metric == 'r2':
                scores = [r2_score(orig[ch], pred[ch]) for ch in range(n_ch)]
            elif metric == 'corr':
                scores = [np.corrcoef(orig[ch], pred[ch])[0, 1] for ch in range(n_ch)]
            else:
                raise ValueError("metric must be 'r2' or 'corr'")

            fit_accumulator += np.asarray(scores)

        # 5) Average over epochs and return dict
        fit_accumulator /= n_epochs
        return dict(zip(epochs_ref.ch_names, fit_accumulator))

    # ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Example usage for subject 101, condition Coordination
    recon = EEGSourceReconstruction(
        source_path="data/source_reconstruction/",
        subjects_dir="~/mne_data/MNE-fsaverage-data/",
        montage="standard_1005",
        method="dSPM",
        loose=1.0,
        depth=0.8,
        snr=3.0,
        verbose=True,
        log_file="source_reconstruction.txt"
    )

    # Load EEGLAB epochs
    recon.load_epochs_eeglab(filename="data/raw/PPT1/s_101_Coordination.set")

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

    # After you have epochs_, fwd_, inverse_operator_ ready
    fit = recon.assess_data_fit(pick_ori='vector', metric='r2')

    # Quick view: print best & worst channels
    best = sorted(fit.items(), key=lambda kv: kv[1], reverse=True)[:5]
    worst = sorted(fit.items(), key=lambda kv: kv[1])[:5]
    print("Best-explained sensors:", best)
    print("Worst-explained sensors:", worst)

    # Save the epochs array
    recon.save_epochs_array(
        data_3d=label_ts_reshaped,
        label_names=label_names,
        out_fname='s_101_Coordination-source-epo.fif'
    )

    # Quick plot noise covariance
    # recon.quick_plot_noise_cov()
    # Quick plot sensor layout
    recon.quick_plot_sensors()
    # Quick plot alignment
    recon.quick_plot_alignment()
    # Suppose you reloaded your ROI-level epochs into roi_epochs:
    roi_epochs = mne.read_epochs("data/source_reconstruction/s_101_Coordination-source-epo.fif")

    # Then call the new plotting method, e.g. for epoch index 4
    recon.plot_sensor_vs_roi_epochs(
        sensor_epochs=recon.epochs_,
        roi_epochs=roi_epochs,
        sensor_name='P1',                     # or any valid sensor in recon.epochs_.ch_names
        roi_name='superiorparietal-lh',       # or any ROI label in roi_epochs.ch_names
        invert_roi=False)