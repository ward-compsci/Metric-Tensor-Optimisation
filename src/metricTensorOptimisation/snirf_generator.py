import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.special import gamma

from snirf import Snirf, validateSnirf
import os

class snirfGenerator:
    def __init__(self, channel_values, duration=60, display_fs=8.9, stim_events=None):
        self.num_channels = len(channel_values)
        self.channel_activations = np.array(channel_values)
        self.duration = duration
        self.generation_fs = display_fs * 100
        self.generation_t = np.arange(0,duration, 1/self.generation_fs)
        self.display_fs = display_fs
        self.display_t = np.arange(0,duration, 1/display_fs)
        self.signals = {channel: None for channel in range(self.num_channels)}
        self.channel_haemodynamics = np.zeros([int(duration * self.generation_fs), self.num_channels, 2]) # combined measurements (conc space)
        self.channel_haemodynamics_split = np.zeros([int(duration * self.generation_fs), self.num_channels, 2, 4]) # split the measurements into components

        self.stimulus_events = stim_events or [(10, 2)]

        self.convolve_and_scale_hrf()

        self.add_channel_noise()

        self.concentration_to_intensity()

        self.create_file()
        self.populate_file()
        self.close_save_file()

    def create_file(self):
        os.makedirs('output', exist_ok=True)
        self.filepath = 'data\\synthetic_data\\synthetic_dataset.snirf'
        self.snirf = Snirf(self.filepath, 'w')

    def populate_file(self):
        self.snirf.formatVersion = '1.0'
        self.snirf.nirs.appendGroup()
        
        # Fill metaDataTags
        self.snirf.nirs[0].metaDataTags.SubjectID = "Subject01"
        self.snirf.nirs[0].metaDataTags.MeasurementDate = "2025-01-01"
        self.snirf.nirs[0].metaDataTags.MeasurementTime = "12:00:00"
        self.snirf.nirs[0].metaDataTags.LengthUnit = "mm"
        self.snirf.nirs[0].metaDataTags.TimeUnit = "s"
        self.snirf.nirs[0].metaDataTags.FrequencyUnit = "Hz"

        self.snirf.nirs[0].data.appendGroup()

        self.snirf.nirs[0].data[0].dataTimeSeries = np.zeros([len(self.display_t), self.num_channels * 2])

        counter = 0
        for channel in range(self.num_channels):
            self.snirf.nirs[0].data[0].dataTimeSeries[:,counter] = self.detected_light[:,channel,0]
            self.snirf.nirs[0].data[0].dataTimeSeries[:,counter+1] = self.detected_light[:,channel,1]
            
            self.snirf.nirs[0].data[0].measurementList.appendGroup()
            self.snirf.nirs[0].data[0].measurementList.appendGroup()

            self.snirf.nirs[0].data[0].measurementList[counter].sourceIndex = channel
            self.snirf.nirs[0].data[0].measurementList[counter].detectorIndex = channel

            self.snirf.nirs[0].data[0].measurementList[counter+1].sourceIndex = channel
            self.snirf.nirs[0].data[0].measurementList[counter+1].detectorIndex = channel
            
            self.snirf.nirs[0].data[0].measurementList[counter].wavelengthIndex = 1
            self.snirf.nirs[0].data[0].measurementList[counter+1].wavelengthIndex = 2

            self.snirf.nirs[0].data[0].measurementList[counter].dataType = 1
            self.snirf.nirs[0].data[0].measurementList[counter].dataTypeIndex = 0

            self.snirf.nirs[0].data[0].measurementList[counter+1].dataType = 1
            self.snirf.nirs[0].data[0].measurementList[counter+1].dataTypeIndex = 0

            counter += 2


        self.snirf.nirs[0].data[0].time = self.display_t

        self.snirf.nirs[0].stim.appendGroup()
        self.snirf.nirs[0].stim[0].name = 'Stim01'
        # self.snirf.nirs[0].stim[0].data = np.array([[self.stimulus_start,self.stimulus_duration,1]]) # [starttime duration value]
 
        stim_array = np.array([[start, dur, 1] for start, dur in self.stimulus_events])
        self.snirf.nirs[0].stim[0].data = stim_array

        self.snirf.nirs[0].probe.wavelengths = np.array([760,850])
        self.snirf.nirs[0].probe.sourcePos2D = None
        self.snirf.nirs[0].probe.detectorPos2D = None
        
        spacing = 0.03 # metres
        source_positions = np.zeros([self.num_channels, 3])
        detector_positions = np.zeros([self.num_channels, 3])

        for i in range(self.num_channels):
            base_x = i * spacing
            source_positions[i] = [base_x, 0, 0]
            detector_positions[i] = [base_x, 0.025, 0] # 2.5cm apart

        self.snirf.nirs[0].probe.sourcePos3D = source_positions
        self.snirf.nirs[0].probe.detectorPos3D = detector_positions

        self.snirf.nirs[0].probe.detectorLabels = [f"D{i+1}" for i in range(self.num_channels)]
        self.snirf.nirs[0].probe.sourceLabels = [f"S{i+1}" for i in range(self.num_channels)]

    def close_save_file(self):
        # self.snirf.close()
        self.snirf.save(self.filepath)
        result = validateSnirf(self.filepath)
        result.display(severity=3)

    def concentration_to_intensity(self, I_inc_bounds=(1000, 5000)):
        extinction_coeffs_760 = np.array([609.45, 1674.07])*1e-6 # HbO, HbR
        extinction_coeffs_850 = np.array([1159.31, 785.9])*1e-6
        
        downsampled_signal = self.downsample_signal(self.channel_haemodynamics)

        mua_690 = extinction_coeffs_760[0] * (downsampled_signal[:,:,0]) + extinction_coeffs_760[1] * (downsampled_signal[:,:,1])
        mua_830 = extinction_coeffs_850[0] * (downsampled_signal[:,:,0]) + extinction_coeffs_850[1] * (downsampled_signal[:,:,1])

        # If L * mua = log10(I_inc / I_det), then I_det = I_inc / 10^(L * mua)
        # We can set I_inc to some arbitrary number
        
        # I_inc = 1e3
        # Generate I_inc randomly for each (time, channel, wavelength)

        self.detected_light = np.zeros_like(downsampled_signal)

        for channel in range(self.num_channels):
            I_inc = np.random.uniform(I_inc_bounds[0], I_inc_bounds[1])

            I_det_690 = I_inc / 10**(mua_690[:,channel] * 2.5e-2)
            I_det_830 = I_inc / 10**(mua_830[:,channel] * 2.5e-2)

            self.detected_light[:,channel,0] = I_det_690
            self.detected_light[:,channel,1] = I_det_830


        motion = self.create_motion_artifacts(self.detected_light)
        self.motion_artifacts = motion
        self.detected_light += motion
        # self.detected_light = self.create_linear_baseline_shift(self.detected_light)

    def double_gamma(self, A1=1, A2=0.35, n1=6, tau1=1, n2=12, tau2=1):
        h1 = A1 * ((self.generation_t ** (n1 - 1)) * np.exp(-self.generation_t / tau1)) / (tau1 ** n1 * gamma(n1))
        h2 = A2 * ((self.generation_t ** (n2 - 1)) * np.exp(-self.generation_t / tau2)) / (tau2 ** n2 * gamma(n2))
        return h1 - h2
    

    def convolve_and_scale_hrf(self):
        hrf = self.double_gamma()
        stimulus = np.zeros_like(self.generation_t)

        for start, dur in self.stimulus_events:
            onset = int(start * self.generation_fs)
            offset = int((start + dur) * self.generation_fs)
            stimulus[onset:offset] = 1

        convolved_response = np.convolve(stimulus, hrf)[:len(stimulus)]
        convolved_response /= np.max(convolved_response)  # Normalize

        temp = self.channel_activations[:, np.newaxis] * convolved_response
        self.channel_haemodynamics[:, :, 0] = temp.T
        self.channel_haemodynamics[:, :, 1] = self.channel_haemodynamics[:, :, 0] * -1 / 3
        self.channel_haemodynamics_split[:, :, :, 0] = self.channel_haemodynamics
        self.channel_haemodynamics_truth = self.downsample_signal(self.channel_haemodynamics)

    def modulate_signal(self, mean_freq, std):
        """
        Generate a modulated time vector based on random frequency variations.
        """
        max_freq = mean_freq + 2 * std
        min_freq = mean_freq - 2 * std

        max_cycle_time = 1 / max_freq
        max_cycles = int(self.duration / max_cycle_time)
        random_freqs = np.random.uniform(min_freq, max_freq, max_cycles)
        cycle_durations = 1 / random_freqs

        normalised_time_vector = []
        total_time = 0

        for cycle_duration in cycle_durations:
            num_samples = int(cycle_duration * self.generation_fs)
            normalised_time_for_cycle = np.linspace(0, 1, num_samples)
            normalised_time_vector.extend(normalised_time_for_cycle)
            total_time += (num_samples / self.generation_fs)
            if total_time > self.duration:
                break

        normalised_time_vector = np.asarray(normalised_time_vector)

        # Adjust the length of the vector to match self.generation_fs * self.duration
        target_length = int(self.generation_fs * self.duration)
        if len(normalised_time_vector) > target_length:
            return normalised_time_vector[:target_length]
        else:
            # Pad the signal to the required length
            return np.pad(normalised_time_vector, (0, target_length - len(normalised_time_vector)), mode='wrap')

    def cardiac_signal(self, mean, std):
        
        def blood_pressure_cycle(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
            return a1 * np.sin(b1 * x + c1) + a2 * np.sin(b2 * x + c2) + a3 * np.sin(b3 * x + c3) + a4 * np.sin(b4 * x + c4)

        def apply_lag_and_smooth(signal, lag_seconds, smoothing_window):
            """
            Apply a lag and smooth the signal to simulate venous dynamics.
            """
            lag_samples = int(self.generation_fs * lag_seconds)
            smoothed_signal = uniform_filter1d(np.roll(signal, lag_samples), size=int(self.generation_fs * smoothing_window))
            return smoothed_signal

        # Fitted blood pressure signal - it just works
        x_points = np.array([3.063, 3.927, 5.183, 6.283, 7.068, 7.696, 8.482, 9.11, 9.895, 10.838, 12.251, 12.88, 13.508, 14.136, 14.921, 15.707, 17.277, 18.848, 19.948, 20.89, 21.675, 22.461, 23.717, 25.131, 26.152])
        y_points = np.array([1.104, 1.534, 2.27, 3.067, 4.356, 5.46, 6.503, 7.607, 8.405, 8.896, 8.466, 7.853, 6.994, 6.258, 5.583, 4.908, 4.663, 4.417, 3.742, 3.129, 2.699, 2.025, 1.656, 1.227, 1.104])
        x_points = (x_points - min(x_points)) / (max(x_points) - min(x_points))
        y_points = (y_points - min(y_points)) / (max(y_points) - min(y_points))
        p0 = [4.4842, 0.0398, 0.6950, 1.0875, 0.5338, 2.3128, 0.5311, 0.8112, -0.9401, 3.4632, 0.2433, -1.3235]
        popt, _ = curve_fit(blood_pressure_cycle, x_points, y_points, p0=p0, maxfev=100000)

        blood_pressure = blood_pressure_cycle(self.modulate_signal(mean, std), *popt)

        cardiac_oxy = (blood_pressure - 0.5) * 2
        cardiac_deoxy = apply_lag_and_smooth(cardiac_oxy, lag_seconds=0.1, smoothing_window=0.1)

        return np.stack([cardiac_oxy, cardiac_deoxy]).T

    def generate_oscillations(self, time_vector, scaling_factor):
        """
        Generate sinusoidal oscillations using the provided time vector.
        """
        phase = time_vector * 2 * np.pi
        return scaling_factor * np.sin(phase)


    def generate_signal(self, mean_freq, std_dev, scaling_factor):
        """
        Generate a modulated sinusoidal signal
        """
        signal = np.zeros([int(self.generation_fs * self.duration),2])
        time_vector = self.modulate_signal(mean_freq, std_dev)
        signal[:,0] = self.generate_oscillations(time_vector, scaling_factor)
        signal[:,1] = signal[:,0] * -0.5
        return signal
    

    def add_channel_noise(self):
        """
        Add the different physiological noises together.
        """
        noise_param_scaling = 3
        signal_params = {
            "cardiac": {"mean_freq": 1.08, "std_dev": 0.16, "scaling_factor": 1.0 * noise_param_scaling},
            "breathing": {"mean_freq": 0.22, "std_dev": 0.07, "scaling_factor": 0.3 * noise_param_scaling},
            "low_freq": {"mean_freq": 0.082, "std_dev": 0.016, "scaling_factor": 0.6 * noise_param_scaling},
        }

        for channel in range(self.num_channels):
            cardiac_noise = self.cardiac_signal(signal_params["cardiac"]["mean_freq"], signal_params["cardiac"]["std_dev"])
            breathing_noise = self.generate_signal(signal_params["breathing"]["mean_freq"],signal_params["breathing"]["std_dev"],signal_params["breathing"]["scaling_factor"])
            low_freq_noise = self.generate_signal(signal_params["low_freq"]["mean_freq"],signal_params["low_freq"]["std_dev"],signal_params["low_freq"]["scaling_factor"])

            self.channel_haemodynamics_split[:,channel,:,1] = cardiac_noise
            self.channel_haemodynamics_split[:,channel,:,2] = breathing_noise
            self.channel_haemodynamics_split[:,channel,:,3] = low_freq_noise

            self.channel_haemodynamics[:,channel,:] += cardiac_noise + breathing_noise + low_freq_noise


    def create_motion_artifacts(self, signals, num_spikes=5, num_shifts=0, spike_intensity_range=(1, 1.5),
                                    spike_duration=1, shift_magnitude_range=(-1.3,1.3)):
            
            signals_with_motion = np.copy(signals)
            # spike_times = np.random.randint(0, len(self.display_t)-spike_duration,num_spikes)
            shift_times = np.random.randint(0, len(self.display_t)-1, num_shifts)
            
            for i in range(2):
                for shift_time in shift_times:
                    for channel in range(self.num_channels):
                        shift_value = np.random.uniform(*shift_magnitude_range) * np.max(np.abs(signals[:,channel,i]))
                        signals_with_motion[shift_time:,channel,i] += shift_value
            
            for channel in range(self.num_channels):
                spike_times = np.random.randint(0, len(self.display_t)-spike_duration,num_spikes)
                for spike_time in spike_times:
                    for w in range(2):
                        spike_intensity = np.random.uniform(*spike_intensity_range) * np.max(np.abs(signals_with_motion[:,channel,w]))
                        spike = np.zeros(spike_duration)
                        spike[spike_duration // 2] = spike_intensity
                        signals_with_motion[spike_time:spike_time+spike_duration,channel,w] =+ spike

            # for i in range(2):
            #     for spike_time in spike_times:
            #         for channel in range(self.num_channels):
            #             spike_intensity = np.random.uniform(*spike_intensity_range) * np.max(np.abs(signals_with_motion[:,channel,i]))
            #             spike = np.zeros(spike_duration)
            #             spike[spike_duration // 2] = spike_intensity
            #             signals_with_motion[spike_time:spike_time+spike_duration,channel,i] =+ spike

            self.motion_noise = signals_with_motion
            return signals_with_motion
   
    def create_linear_baseline_shift(self, signals, baseline_range=(0, 0.2)):
        
        signals_with_baseline = np.copy(signals)
        timepoints = signals.shape[0]
        time_vector = np.linspace(0, 1, timepoints)  # Normalized time vector
        
        for i in range(signals.shape[2]):  # Loop over modalities (e.g., oxygenated/deoxygenated)
            for channel in range(signals.shape[1]):  # Loop over channels
                slope = np.random.uniform(*baseline_range) * np.max(np.abs(signals[:,channel,i])) # Draw a slope from the range
                baseline_shift = slope * time_vector  # Create the linear shift
                signals_with_baseline[:, channel, i] += baseline_shift
        
        return signals_with_baseline

    def downsample_signal(self, signal):
        block_size = int(self.generation_fs / self.display_fs)
        num_samples = signal.shape[0]
        num_blocks = num_samples // block_size

        downsampled_signal = np.zeros((num_blocks, signal.shape[1], signal.shape[2]))
        for ch in range(signal.shape[1]):  # Loop over channels
            for dim in range(signal.shape[2]):  # Loop over oxygenated and deoxygenated signals
                reshaped_data = signal[:num_blocks * block_size, ch, dim].reshape(num_blocks, block_size)
                downsampled_signal[:, ch, dim] = np.mean(reshaped_data, axis=1)
        
        return downsampled_signal

    def display(self):
        downsampled_signal = self.downsample_signal(self.channel_haemodynamics)
        fig,axes = plt.subplots(self.num_channels,2)
        for i in range(self.num_channels):
            axes[i,0].plot(self.display_t, downsampled_signal[:,i,0])
            axes[i,1].plot(self.display_t, downsampled_signal[:,i,1])
        
        fig,axes = plt.subplots(self.num_channels,2,sharey=True)
        for i in range(self.num_channels):
            axes[i,0].plot(self.display_t, self.detected_light[:,i,0])
            axes[i,1].plot(self.display_t, self.detected_light[:,i,1])

        plt.show()

    def plot_channel_noise_and_combinations(self, channel_index=0):
        """
        Plot the data for one channel using `self.channel_haemodynamics_split`.
        The right column will include the HRF and HRF + noise.
        """
        # Define noise types and combinations
        noise_types = ["cardiac", "breathing", "low_freq"]
        noise_combinations = [
            ["cardiac", "breathing"],
            ["cardiac", "low_freq"],
            ["breathing", "low_freq"],
            ["cardiac", "breathing", "low_freq"],
        ]


        temp = []
        for i in range(4):
            temp.append(self.downsample_signal(self.channel_haemodynamics_split[:,:,:,i]))

        channel_measurements_split_downsampled = np.asarray(temp)
        channel_measurements_split_downsampled = channel_measurements_split_downsampled[:,:,:,:]

        # Create figure
        num_rows = len(noise_types) + len(noise_combinations) + 1  # Add 1 for HRF-only row
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 5 * num_rows), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.2]}, layout='constrained')
        # 0 noise
        axes[0, 0].plot([self.generation_t[0], self.generation_t[-1]], [0, 0], label='HbO', color='r')
        axes[0, 0].plot([self.generation_t[0], self.generation_t[-1]], [0, 0], label='HbR', color='b')
        axes[0, 0].set_title('Noise')

        # Plot HRF-only row
        row = 0
        # hrf_signal = self.channel_haemodynamics_split[:, channel_index, :, 0]
        hrf_signal = channel_measurements_split_downsampled[0,:,channel_index,:]
        axes[row, 1].plot(self.display_t, hrf_signal[:, 0], label="HbO", color='r')
        axes[row, 1].plot(self.display_t, hrf_signal[:, 1], label="HbR", color='b')
        axes[row, 1].set_title("HRF + Noise")
        axes[row, 1].legend()

        axes[row, 2].axis("off")

        # Plot individual noise types
        for i, noise_type in enumerate(noise_types):
            row += 1
            noise_signal = channel_measurements_split_downsampled[i+1,:,channel_index,:]
            combined_signal = noise_signal + hrf_signal

            # Left: noise only
            axes[row, 0].plot(self.display_t, noise_signal[:, 0], label="HbO", color='r')
            axes[row, 0].plot(self.display_t, noise_signal[:, 1], label="HbR", color='b')
            axes[row, 0].legend()

            # Right: HRF + noise
            axes[row, 1].plot(self.display_t, combined_signal[:, 0], label="HbO", color='r')
            axes[row, 1].plot(self.display_t, combined_signal[:, 1], label="HbR", color='b')
            axes[row, 1].legend()

            # Label column
            axes[row, 2].text(0.5, 0.5, noise_type.capitalize(), va="center", ha="center", rotation=0, fontsize=12)
            axes[row, 2].axis("off")

        # Plot noise combinations
        for combination in noise_combinations:
            row += 1
            noise_indices = [noise_types.index(noise) + 1 for noise in combination]
            combined_noise = sum(
                channel_measurements_split_downsampled[idx, :, channel_index, :]
                for idx in noise_indices
            )
            combined_signal = combined_noise + hrf_signal

            # Left: noise only
            axes[row, 0].plot(self.display_t, combined_noise[:, 0], label="HbO", color='r')
            axes[row, 0].plot(self.display_t, combined_noise[:, 1], label="HbR", color='b')
            axes[row, 0].legend()

            # Right: HRF + noise
            axes[row, 1].plot(self.display_t, combined_signal[:, 0], label="HbO", color='r')
            axes[row, 1].plot(self.display_t, combined_signal[:, 1], label="HbR", color='b')
            axes[row, 1].legend()


            # Label column
            label_text = "\n".join([n.capitalize() for n in combination])
            axes[row, 2].text(0.5, 0.5, label_text, va="center", ha="center", rotation=0, fontsize=12)
            axes[row, 2].axis("off")


        axes[-1,0].set_xlabel('Time [s]')
        axes[-1,1].set_xlabel('Time [s]')

        fig.supylabel('Amplitude [A.U.]')

        plt.show()

    def plot_channel_intensity(self):
        
        fig,axes = plt.subplots(self.num_channels,3,gridspec_kw={"width_ratios": [1, 1, 0.2]}, layout='constrained')
        for i in range(self.num_channels):
            axes[i,0].plot(self.display_t,self.detected_light[:,i,0], color='tab:orange') # wv1
            axes[i,1].plot(self.display_t,self.detected_light[:,i,1], color='tab:cyan') # wv2

            axes[i,0].set_ylim([1980,2020])
            axes[i,1].set_ylim([1980,2020])

            if self.channel_activations[i] == 0:
                axes[i,2].text(0.5, 0.5, f'Channel {i}:\nNo\nActivation', va="center", ha="center", rotation=0, fontsize=20, weight='bold')
            else:
                axes[i,2].text(0.5, 0.5, f'Channel {i}:\nactivation', va="center", ha="center", rotation=0, fontsize=20, weight='bold')
            axes[i,2].axis("off")


        # Set tick sizes
        for ax in axes[:, :2].flatten():  # Select only the first two columns
            ax.tick_params(axis='both', which='major', labelsize=20)  # Adjust label size

            # ax.set_xticklabels(ax.get_xticks(), fontweight='bold')
            # ax.set_yticklabels(ax.get_yticks(), fontweight='bold')


        axes[0,0].set_title('690nm',fontsize=20, weight='bold')
        axes[0,1].set_title('830nm',fontsize=20, weight='bold')
        fig.supylabel('Amplitude [A.U.]',fontsize=20, weight='bold')
        fig.supxlabel('Time [s]',fontsize=20, weight='bold')

        plt.rcParams.update(plt.rcParamsDefault) 

        plt.show()

    def plot_channel_intensity_with_hrf(self):
            
            fig,axes = plt.subplots(self.num_channels,4,gridspec_kw={"width_ratios": [1, 1, 0.2, 1]}, layout='constrained')
            for i in range(self.num_channels):
                axes[i,0].plot(self.display_t,self.detected_light[:,i,0], color='tab:orange') # wv1
                axes[i,1].plot(self.display_t,self.detected_light[:,i,1], color='tab:cyan') # wv2

                # axes[i,0].set_ylim([1980,2020])
                # axes[i,1].set_ylim([1980,2020])

                if self.channel_activations[i] == 0:
                    axes[i,2].text(0.5, 0.5, f'Channel {i}:\nNo\nActivation', va="center", ha="center", rotation=0, fontsize=20, weight='bold')
                else:
                    axes[i,2].text(0.5, 0.5, f'Channel {i}:\nactivation', va="center", ha="center", rotation=0, fontsize=20, weight='bold')
                axes[i,2].axis("off")

                axes[i,3].plot(self.display_t, self.channel_haemodynamics_truth[:,i,0],color='red')
                axes[i,3].plot(self.display_t, self.channel_haemodynamics_truth[:,i,1],color='blue')


            # Set tick sizes
            for ax in axes[:, :2].flatten():  # Select only the first two columns
                ax.tick_params(axis='both', which='major', labelsize=20)  # Adjust label size

            for ax in axes[:, 3].flatten():
                ax.tick_params(axis='both', which='major', labelsize=20)  # Adjust label size


            axes[0,0].set_title('690nm',fontsize=20, weight='bold')
            axes[0,1].set_title('830nm',fontsize=20, weight='bold')
            fig.supylabel('Amplitude [A.U.]',fontsize=20, weight='bold')
            fig.supxlabel('Time [s]',fontsize=20, weight='bold')

            plt.rcParams.update(plt.rcParamsDefault) 

            plt.show()


    def plot_different_noise_types(self, channel_index=0):
        """
        Plot the data for one channel using `self.channel_haemodynamics_split`.
        The right column will include the HRF and HRF + noise.
        """
        # Define noise types and combinations
        noise_types = ["cardiac", "breathing", "low_freq"]
        noise_combinations = [
            ["cardiac", "breathing"],
            ["cardiac", "low_freq"],
            ["breathing", "low_freq"],
            ["cardiac", "breathing", "low_freq"],
        ]


        temp = []
        for i in range(4):
            temp.append(self.downsample_signal(self.channel_haemodynamics_split[:,:,:,i]))

        channel_measurements_split_downsampled = np.asarray(temp)
        channel_measurements_split_downsampled = channel_measurements_split_downsampled[:,:,:,:]

        from matplotlib import rcParams
        rcParams['font.family'] = 'Times New Roman'

        # Create figure
        num_rows = len(noise_types) + 3  # Add 1 for HRF-only row, 1 for motion, 1 for final light
        fig, axes = plt.subplots(num_rows, 3, figsize=(6, 4), sharex=True, constrained_layout=True)
        # 0 noise
        axes[0, 0].plot([self.generation_t[0], self.generation_t[-1]], [0, 0], label='HbO', color='r',linewidth=0.5)
        axes[0, 0].plot([self.generation_t[0], self.generation_t[-1]], [0, 0], label='HbR', color='b',linewidth=0.5)
        # axes[0, 0].set_title('Noise')
        axes[0, 0].legend(fontsize=6,loc="upper right")


        # Plot HRF-only row
        row = 0
        # hrf_signal = self.channel_haemodynamics_split[:, channel_index, :, 0]
        hrf_signal = channel_measurements_split_downsampled[0,:,channel_index,:]
        axes[row, 1].plot(self.display_t, hrf_signal[:, 0], label="HbO", color='r',linewidth=0.5)
        axes[row, 1].plot(self.display_t, hrf_signal[:, 1], label="HbR", color='b',linewidth=0.5)
        # axes[row, 1].set_title("HRF + Noise")

        axes[row, 2].axis("off")
        axes[row, 2].text(0.5, 0.5, "Initial", va="center", ha="center", rotation=0, fontsize=8)


        # Plot individual noise types
        for i, noise_type in enumerate(noise_types):
            row += 1
            noise_signal = channel_measurements_split_downsampled[i+1,:,channel_index,:]
            combined_signal = noise_signal + hrf_signal

            # Left: noise only
            axes[row, 0].plot(self.display_t, noise_signal[:, 0], label="HbO", color='r',linewidth=0.5)
            axes[row, 0].plot(self.display_t, noise_signal[:, 1], label="HbR", color='b',linewidth=0.5)
            # axes[row, 0].legend(fontsize=10)

            # Right: HRF + noise
            axes[row, 1].plot(self.display_t, combined_signal[:, 0], label="HbO", color='r',linewidth=0.5)
            axes[row, 1].plot(self.display_t, combined_signal[:, 1], label="HbR", color='b',linewidth=0.5)
            # axes[row, 1].legend(fontsize=10)

            # Label column
            axes[row, 2].text(0.5, 0.5, noise_type.capitalize(), va="center", ha="center", rotation=0, fontsize=8)
            axes[row, 2].axis("off")

        axes[-2, 0].plot(self.display_t, self.motion_artifacts[:, 1, 0], color='tab:orange',label="760nm",linewidth=0.5)
        axes[-2, 0].plot(self.display_t, self.motion_artifacts[:, 1, 1], color='tab:blue',label="850nm",linewidth=0.5)
        axes[-2, 0].legend(fontsize=6,loc="center left")

        axes[-2, 1].plot(self.display_t, self.motion_artifacts[:, 0, 0], color='tab:orange',label="760nm",linewidth=0.5)
        axes[-2, 1].plot(self.display_t, self.motion_artifacts[:, 0, 1], color='tab:blue',label="850nm",linewidth=0.5)
        axes[-2, 2].text(0.5, 0.5, "Motion", va="center", ha="center", rotation=0, fontsize=8)
        # axes[-2, 2].legend(fontsize=10)

        axes[-2, 2].axis("off")

        axes[-1, 0].plot(self.display_t, self.detected_light[:, 1, 0], color='tab:orange',label="760nm",linewidth=0.5)
        axes[-1, 0].plot(self.display_t, self.detected_light[:, 1, 1], color='tab:blue',label="850nm",linewidth=0.5)
        axes[-1, 0].set_ylim(3071, 3073)
        # axes[-1, 0].legend(fontsize=10)


        axes[-1, 1].plot(self.display_t, self.detected_light[:, 0, 0], color='tab:orange',label="760nm",linewidth=0.5)
        axes[-1, 1].plot(self.display_t, self.detected_light[:, 0, 1], color='tab:blue',label="850nm",linewidth=0.5)
        axes[-1, 1].set_ylim(5226, 5230)
        axes[-1, 2].text(0.5, 0.5, "Final", va="center", ha="center", rotation=0, fontsize=8)
        # axes[-1, 1].legend(fontsize=10)

        axes[-1, 2].axis("off")

        for ax in axes.flatten():
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)

        fig.supylabel("Amplitude (A.U.)", fontsize=8)
        # fig.supxlabel("Time (s)")
        axes[-1,0].set_xlabel("Time (s)", fontsize=8)
        axes[-1,1].set_xlabel("Time (s)", fontsize=8)



        plt.savefig("synthetic_noise.pdf")

        plt.show()

# if __name__ = "__main__":
#     # Example Usage
#     import numpy as np
#     np.random.seed(42)

#     ground_truth = np.array([1,0,1,0,1])
#     snirf_file = snirfGenerator(ground_truth)
#     snirf_file.plot_different_noise_types()
