import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import KMeans
import mne

class LEiDAEEGAnalyzer:
    """
    A class to perform the LEiDA (Leading Eigenvector Dynamics Analysis) pipeline on EEG data.

    The pipeline includes:
      1) Per-channel bandpass filtering.
      2) Hilbert transform to extract instantaneous phases.
      3) Dynamic phase-locking (dPL) matrix computation in fixed windows.
      4) Leading eigenvector extraction for each window.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    freq_band : str
        Which frequency band to analyze: 'alpha', 'beta', or 'gamma'.
    window_size : int
        Number of samples in each non-overlapping window (e.g., 250).
    remove_edges : bool
        If True, skip the first and last window (as done in some MATLAB code).
        Otherwise, keep them.
    do_plots : bool
        If True, show optional diagnostic plots.
    verbose : bool
        If True, print progress messages.
    
    Methods
    -------
    set_frequency_band(freq_band)
        Update lowcut/highcut based on 'alpha', 'beta', or 'gamma'.
    filter_data(data)
        Filter each channel of input data using a zero-phase Butterworth filter.
    compute_hilbert_phases(filtered_data)
        Compute instantaneous phases via Hilbert transform (per channel).
    compute_leading_eigenvectors(data_3d)
        Run the LEiDA pipeline over an array of shape (n_epochs, n_channels, n_timepoints).
    _compute_windows(phases)
        Private helper to subdivide phases into windows, build dPL, and compute leading eigenvectors.
    plot_filter_example(original, filtered, fs, epoch_idx=0, channel_idx=0)
        (Optional) Plot an example channel before and after filtering.
    plot_phase_example(phases, fs, epoch_idx=0, channel_idx=0)
        (Optional) Plot an example channel’s phase.
    plot_example_dpl(iFC, V1)
        (Optional) Visualize a dynamic phase-locking matrix and the associated leading eigenvector.
    """

    def __init__(self,
                 fs: float,
                 freq_band: str = 'alpha',
                 window_size: int = 256,
                 remove_edges: bool = True,
                 do_plots: bool = False,
                 verbose: bool = True):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        freq_band : str, optional
            Desired frequency band ('alpha', 'beta', or 'gamma'). Default is 'alpha'.
        window_size : int, optional
            Window size in samples for dPL calculations. Default is 250.
        remove_edges : bool, optional
            If True, skip first and last window. Default is True.
        do_plots : bool, optional
            Whether to generate diagnostic plots. Default is False.
        verbose : bool, optional
            Whether to print status messages. Default is True.
        """
        self.fs = fs
        self.freq_band = freq_band
        self.window_size = window_size
        self.remove_edges = remove_edges
        self.do_plots = do_plots
        self.verbose = verbose

        # Track whether we've already shown each plot type
        self.did_plot_filter = False
        self.did_plot_phase = False
        self.did_plot_dpl = False

        # Initialize frequency band limits
        self.lowcut, self.highcut = 8, 12  # alpha defaults
        self.set_frequency_band(freq_band)

    def set_frequency_band(self, freq_band: str):
        """
        Set the lowcut/highcut frequency range based on the chosen freq_band.

        Parameters
        ----------
        freq_band : str
            Either 'alpha', 'beta', or 'gamma'.
        """
        if freq_band == 'alpha':
            self.lowcut, self.highcut = 8, 12
        elif freq_band == 'beta':
            self.lowcut, self.highcut = 15, 25
        elif freq_band == 'gamma':
            self.lowcut, self.highcut = 30, 80
        else:
            raise ValueError("freq_band must be 'alpha', 'beta', or 'gamma'.")
        self.freq_band = freq_band

        if self.verbose:
            print(f"Frequency band set to {freq_band} "
                  f"({self.lowcut}-{self.highcut} Hz).")

    def _butter_bandpass(self, lowcut: float, highcut: float, fs: float, order=6):
        """
        Construct bandpass filter coefficients for a Butterworth filter.
        """
        from scipy.signal import butter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase bandpass filtering to each channel of the input.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_timepoints)
            Single epoch of EEG data or a single trial.

        Returns
        -------
        filtered_data : ndarray, shape (n_channels, n_timepoints)
            The filtered data per channel.
        """
        b, a = self._butter_bandpass(self.lowcut, self.highcut, self.fs, order=6)
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            # De-mean each channel first
            demeaned = data[ch, :] - np.mean(data[ch, :])
            filtered_data[ch, :] = filtfilt(b, a, demeaned)
        return filtered_data

    def compute_hilbert_phases(self, filtered_data: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phases using the Hilbert transform.

        Parameters
        ----------
        filtered_data : ndarray, shape (n_channels, n_timepoints)
            The bandpass filtered EEG data.

        Returns
        -------
        phases : ndarray, shape (n_channels, n_timepoints)
            Phases derived from Hilbert transform.
        """
        # Axis=1 means we're applying Hilbert transform along the time dimension
        analytic_signal = hilbert(filtered_data, axis=1)
        phases = np.angle(analytic_signal)
        return phases

    def _compute_windows(self, phases: np.ndarray, epoch_idx: int) -> np.ndarray:
        """
        Subdivide `phases` into non-overlapping windows of length `self.window_size`,
        build the dynamic phase-locking (dPL) matrix for each window,
        and return its leading eigenvector (unit-normalised, no sign flip).
        """
        n_chan, T = phases.shape
        W = self.window_size

        # indices of LEFT edges of every complete window -----------------
        starts = np.arange(0, T - W + 1, W)          # e.g. [0,256,512,...]
        if len(starts) == 0:                         # epoch shorter than one window
            return np.empty((0, n_chan))

        if self.remove_edges and len(starts) >= 3:   # need ≥3 windows to drop edges
            starts = starts[1:-1]                    # drop first & last

        lead_eigs = []
        for s in starts:
            seg = phases[:, s:s+W]                   # (n_chan, W)

            # ---------- vectorised iFC  (no for-loops) -----------------
            # pairwise phase-diff tensor: (n_chan, n_chan, W)
            dphase = seg[:, None, :] - seg[None, :, :]
            iFC = np.cos(dphase).mean(axis=-1)       # (n_chan, n_chan)

            # ---------- leading eigenvector ----------------------------
            vals, vecs = np.linalg.eigh(iFC)         # ascending λ
            v1 = vecs[:, -1]                         # largest eigenvalue
            v1 /= np.linalg.norm(v1)                 # unit length
            lead_eigs.append(v1)

            # optional one-off plot
            if (
                self.do_plots
                and not self.did_plot_dpl
                and epoch_idx == 0
                and len(lead_eigs) == 11             # e.g. 10th window (0-based)
            ):
                self.plot_example_dpl(iFC, v1)
                self.did_plot_dpl = True

        return np.vstack(lead_eigs)                  # (n_windows, n_chan)


    def compute_leading_eigenvectors(self, data_3d: np.ndarray) -> np.ndarray:
        """
        Run the LEiDA pipeline over an array of shape (n_epochs, n_channels, n_timepoints).
        
        Parameters
        ----------
        data_3d : ndarray, shape (n_epochs, n_channels, n_timepoints)
            The EEG data to analyze. Each epoch is [n_channels, n_timepoints].
        
        Returns
        -------
        all_lead_eigs : ndarray, shape (n_epochs, n_windows, n_channels)
            Leading eigenvectors for each epoch and each window.
            If remove_edges=True, then n_windows = (#windows_of_epoch - 2) for each epoch.
        """
        n_epochs, n_channels, n_timepoints = data_3d.shape

        if self.verbose:
            print(f"Processing data with shape (epochs={n_epochs}, channels={n_channels}, "
                  f"timepoints={n_timepoints})")
            print(f"Window size = {self.window_size}, removing edges = {self.remove_edges}")
            print(f"Bandpass from {self.lowcut} to {self.highcut} Hz. Order=6")

        epoch_eig_list = []

        for ep_idx in range(n_epochs):
            epoch_data = data_3d[ep_idx, :, :]  # shape (n_channels, n_timepoints)

            # 1) Filter
            filtered_epoch = self.filter_data(epoch_data)

            # Plot filter example once (epoch=0, channel=0)
            if self.do_plots and not self.did_plot_filter and ep_idx == 0:
                self.plot_filter_example(epoch_data[0, :], filtered_epoch[0, :], self.fs, epoch_idx=0, channel_idx=0)
                self.did_plot_filter = True

            # 2) Hilbert phases
            phases = self.compute_hilbert_phases(filtered_epoch)

            # Plot phase example once (epoch=0, channel=0)
            if self.do_plots and not self.did_plot_phase and ep_idx == 0:
                self.plot_phase_example(phases, self.fs, epoch_idx=0, channel_idx=0)
                self.did_plot_phase = True

            # 3) Windowing & dynamic phase-locking + leading eigenvectors
            lead_eigs = self._compute_windows(phases, ep_idx)
            epoch_eig_list.append(lead_eigs)
            # print every 10 epochs
            if self.verbose and ep_idx % 10 == 0:
                print(f"Epoch {ep_idx}/{n_epochs} processed.")

        # Convert list of arrays to a single 3D array
        all_lead_eigs = np.stack(epoch_eig_list, axis=0)

        if self.verbose:
            print(f"\nCompleted LEiDA!!! Output shape = {all_lead_eigs.shape} "
                  "(epochs x windows x channels).")

        return all_lead_eigs

    # ----------------------- PLOTTING ROUTINES -----------------------
    def plot_filter_example(self, original, filtered, fs, epoch_idx=0, channel_idx=0):
        """
        Plot an example channel before and after filtering.

        Parameters
        ----------
        original : ndarray, shape (n_timepoints,)
            Original (demeaned) time series data for one channel.
        filtered : ndarray, shape (n_timepoints,)
            Filtered time series.
        fs : float
            Sampling frequency.
        epoch_idx : int, optional
            Epoch index (for labeling).
        channel_idx : int, optional
            Channel index (for labeling).
        """
        t = np.arange(len(original)) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, original, label='Raw (demeaned)', alpha=0.7)
        plt.plot(t, filtered, label='Filtered', alpha=0.7)
        plt.xlim([0, min(10.0, t[-1])])  # zoom in up to 10s
        plt.legend()
        plt.title(f"Epoch {epoch_idx}, Channel {channel_idx}: Before/After Filtering")
        plt.xlabel("Time (s)")
        plt.show()

    def plot_phase_example(self, phases, fs, epoch_idx=0, channel_idx=0):
        """
        Plot an example channel's instantaneous phase.

        Parameters
        ----------
        phases : ndarray, shape (n_channels, n_timepoints)
            Phase array from Hilbert transform.
        fs : float
            Sampling frequency.
        epoch_idx : int, optional
            Epoch index (for labeling).
        channel_idx : int, optional
            Channel index (for labeling).
        """
        t = np.arange(phases.shape[1]) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, phases[channel_idx, :], label=f'Phase (Ch={channel_idx})')
        plt.title(f"Epoch {epoch_idx}, Channel {channel_idx}: Instantaneous Phase")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (radians)")
        plt.xlim([0, min(10.0, t[-1])])
        plt.legend()
        plt.show()

    def plot_example_dpl(self, iFC, V1):
        """
        Plot a dynamic phase-locking (dPL) matrix and its leading eigenvector.

        Parameters
        ----------
        iFC : ndarray, shape (n_channels, n_channels)
            The phase-locking matrix for a given window.
        V1 : ndarray, shape (n_channels,)
            The leading eigenvector of iFC.
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(iFC, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Phase Coherence (mean cos(diff))')
        plt.title("Example Dynamic Phase-Locking Matrix (dPL)")
        plt.xlabel("Brain Region (ROI index)")
        plt.ylabel("Brain Region (ROI index)")
        plt.show()

        plt.figure(figsize=(6, 4))
        markerline, stemlines, baseline = plt.stem(np.arange(len(V1)), V1)
        plt.setp(markerline, marker='o', markersize=6, color='b')
        plt.setp(stemlines, color='b')
        plt.title("Leading Eigenvector (with sign)")
        plt.xlabel("Brain Region (ROI index)")
        plt.ylabel("Eigenvector Component")
        plt.show()

# ------------------ Main ------------------#
if __name__ == "__main__":
    
    epochs = mne.read_epochs("data/source/s_101_Coordination-source-epo.fif")
    data = epochs.get_data()
    print(f"Data shape: {data.shape}")  # Data shape: (87 epochs, 68 channels , 1280 samples)
    fs = epochs.info['sfreq']
    print(f"Sampling frequency: {fs} Hz")
    window_size = int(fs / 8)  # e.g. 125 ms window

    # Instantiate the analyzer
    leida = LEiDAEEGAnalyzer(fs=fs,
                             freq_band='alpha',
                             window_size=window_size,
                             remove_edges=False,
                             do_plots=True,
                             verbose=True)

    # Compute leading eigenvectors
    all_eigenvectors = leida.compute_leading_eigenvectors(data)
    print("Final shape of leading eigenvectors:", all_eigenvectors.shape)
