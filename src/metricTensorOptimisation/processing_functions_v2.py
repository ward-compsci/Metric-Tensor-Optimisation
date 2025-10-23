import numpy as np
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt
from scipy.stats import linregress
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression

import sys

import matplotlib.pyplot as plt

class signalProcessing:
    def __init__(self,parent):
        self.parent = parent

    def intensity_to_OD(self, signal):
        with np.errstate(divide='ignore', invalid='ignore'):
            OD = np.log10(np.mean(signal,axis=0) / signal)
        return OD

    def OD_to_concentration(self, signal):
        extinction_coeffs_760 = np.array([609.45, 1674.07])*1e-6 # O2Hb, HHb . microM cm^-1
        extinction_coeffs_850 = np.array([1159.31, 785.9])*1e-6
        A = np.vstack([extinction_coeffs_760,extinction_coeffs_850])

        DPF = 6
        distances = np.array([self.parent.channel_distance_dict[ch] * 1e2 * DPF for ch in np.arange(1,len(self.parent.combined_channels)+1)])
        A_inv = np.linalg.pinv([A * distance for distance in distances])  # Shape: (2, C, 2)
        conc_change = np.einsum('cij,tcjp->tcip', A_inv, signal)  # Batch multiplication

        return conc_change

    def butter_lowpass_filter(self, signal, cutoff, order=4):
        sos = butter(order, cutoff / (self.parent.fs / 2), btype='low', output='sos')
        # Create a mask of valid time series (no inf or nan)
        valid_mask = np.all(np.isfinite(signal), axis=0)  # Assumes NaNs/Infs are along time axis
        # Create an output array initialized with NaNs
        filtered_signal = np.full_like(signal, np.nan)
        # Apply filtering only to valid time series
        filtered_signal[:, valid_mask] = sosfiltfilt(sos, signal[:, valid_mask], axis=0)
    
        return filtered_signal
        
    def butter_highpass_filter(self, signal, cutoff, order=4):
        sos = butter(order, cutoff / (self.parent.fs / 2), btype='high', output='sos')
        # Create a mask of valid time series (no inf or nan)
        valid_mask = np.all(np.isfinite(signal), axis=0)
        # Create an output array initialized with NaNs
        filtered_signal = np.full_like(signal, np.nan)
        # Apply filtering only to valid time series
        filtered_signal[:, valid_mask] = sosfiltfilt(sos, signal[:, valid_mask], axis=0)

        return filtered_signal

    def TDDR(self, signal):

        """
        Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
        Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS.
        NeuroImage, 184, 171-179. doi: 10.1016/j.neuroimage.2018.09.025
        """

        data_processed = signal.copy()
        sample_rate = self.parent.fs
        for p in range(np.shape(signal)[3]):
            for ch in range(np.shape(signal)[1]):
                for wv in range(np.shape(signal)[2]):
                    time_series = signal[:,ch,wv,p]
                    
                    if np.isnan(time_series).any() or np.isinf(time_series).any():
                        continue  # Skip this channel if NaN or Inf is found
                    # Preprocess: Separate high and low frequencies
                    filter_cutoff = .5
                    filter_order = 3
                    Fc = filter_cutoff * 2/sample_rate
                    signal_mean = np.mean(time_series)
                    time_series -= signal_mean
                    if Fc < 1:
                        fb, fa = butter(filter_order, Fc)
                        signal_low = filtfilt(fb, fa, time_series, padlen=0)
                    else:
                        signal_low = time_series

                    signal_high = time_series - signal_low
                    # Initialize
                    tune = 4.685
                    D = np.sqrt(np.finfo(time_series.dtype).eps)
                    mu = np.inf
                    iter = 0
                    # Step 1. Compute temporal derivative of the signal
                    deriv = np.diff(signal_low)
                    # Step 2. Initialize observation weights
                    w = np.ones(deriv.shape)
                    # Step 3. Iterative estimation of robust weights
                    while iter < 50:
                        iter = iter + 1
                        mu0 = mu
                        # Step 3a. Estimate weighted mean
                        mu = np.sum(w * deriv) / np.sum(w)
                        # Step 3b. Calculate absolute residuals of estimate
                        dev = np.abs(deriv - mu)
                        # Step 3c. Robust estimate of standard deviation of the residuals
                        sigma = 1.4826 * np.median(dev)

                        if sigma==0:
                            continue
                        if np.isfinite(sigma)==False:
                            sys.exit("Sigma is non-finite")

                        # Step 3d. Scale deviations by standard deviation and tuning parameter
                        r = dev / (sigma * tune)
                        # Step 3e. Calculate new weights according to Tukey's biweight function
                        w = ((1 - r**2) * (r < 1)) ** 2
                        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
                        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
                            break
                    # Step 4. Apply robust weights to centered derivative
                    new_deriv = w * (deriv - mu)
                    # Step 5. Integrate corrected derivative
                    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))
                    # Postprocess: Center the corrected signal
                    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)
                    # Postprocess: Merge back with uncorrected high frequency component
                    signal_corrected = signal_low_corrected + signal_high + signal_mean

                    data_processed[:,ch,wv,p] = signal_corrected

        return data_processed



    def short_channel_regression(self, signal):
        T, C, W, P = signal.shape
        data_corrected = signal.copy()

        for p in range(P):

            for ch in range(len(self.parent.ROI_channels_index_combined)):  # Iterate over long channels

                for w in range(W):  # Iterate over wavelengths
                    short_channel_idx = self.parent.short_channel_indexes[ch, p]# ; print(f"{short_channel_idx=}")
                    X = signal[:,short_channel_idx, w, p].reshape(-1, 1)
                    y = signal[:, ch, w, p]  # Long-channel signal

                    # Check if either X or y contains NaN or Inf values
                    if np.isnan(X).any() or np.isnan(y).any() or np.isinf(X).any() or np.isinf(y).any():
                        continue  # Skip this iteration if NaN or Inf is found
                    
                    # Fit linear regression
                    model = LinearRegression().fit(X, y)
                    
                    # Remove short-channel contribution
                    y_corrected = y - model.predict(X)
                    data_corrected[:, ch, w, p] = y_corrected

        return data_corrected

    
    # def remove_linear_baseline(self, signal):
    #     detrended_signal = signal.copy()

    #     for participant in range(self.parent.num_participants):
    #         for wavelength in range(signal.shape[2]):
    #             for ch in range(signal.shape[1]):
    #                 time_series = signal[:,ch,wavelength,participant]
    #                 if np.sum(1-np.isfinite(time_series)) == 0:
    #                     detrended_signal[:,ch,wavelength,participant] = detrend(time_series,type='linear')
        
    #     return detrended_signal