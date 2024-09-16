import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal
from scipy.signal import find_peaks

# Lowpass and Highpass filter settings
Lowpass_setting = 10  # in Hz
Highpass_setting = 25  # in Hz
window_size = 11  # Window size for moving average
quantile_threshold = 0.7  # Ignore the top 0.5% of the values

# Function to define the bandpass filter
def define_bandpass_filter(lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# Function to apply a differentiation filter
def apply_differentiation_filter(signal):
    return np.diff(signal, n=1)

# Function to apply a squaring filter
def apply_squaring_filter(signal):
    return np.square(signal)

# Function to apply a moving average filter
def apply_moving_average_filter(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Function to determine the threshold based on a quantile
def determine_threshold(signal, quantile):
    threshold_value = np.quantile(signal, quantile)
    top_values = signal[signal >= threshold_value]
    return np.mean(top_values)

# Function to detect peaks, ensuring no overlapping indices
def detect_peaks(signal, threshold, min_distance):
    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
    detected_peaks = []
    
    for peak in peaks:
        if all(abs(peak - p) >= min_distance for p in detected_peaks):  # Check distance from previously detected peaks
            detected_peaks.append(peak)
    
    return np.array(detected_peaks)

# Load the ECG data
record = wfdb.rdrecord(r'C:\Users\baumi\iCloudDrive\Clemson\Computational Modeling\Project 1\CODE\EKG Data\Working Example\e0112')  # Path to your WFDB file
ecg_matrix = np.array(record.p_signal)
sampling_rate = record.fs

# Apply the bandpass filter
b, a = define_bandpass_filter(Lowpass_setting, Highpass_setting, sampling_rate)
filtered_signal = signal.lfilter(b, a, ecg_matrix[:, 0])  # Filter the first channel

# Apply the differentiation filter
Differentiation_signal = apply_differentiation_filter(filtered_signal)

# Apply the squaring filter
Squared_signal = apply_squaring_filter(Differentiation_signal)

# Apply the moving average filter
Moving_average_signal = apply_moving_average_filter(Squared_signal, window_size)

# Determine the threshold using a quantile-based approach
Threshold = determine_threshold(Moving_average_signal, quantile_threshold)

# Apply the threshold: Set values below the threshold to zero
Thresholded_signal = np.where(Moving_average_signal >= Threshold, Moving_average_signal, 0)

# Adjust the distance parameter (tune this depending on your data)
distance_between_peaks = 50  # Set based on expected QRS complex width

# Index-based Peak detection
detected_peaks = detect_peaks(Thresholded_signal, threshold=Threshold, min_distance=distance_between_peaks)

# Only display the first 5000 values for plotting
Thresholded_signal_5000 = Thresholded_signal[:5000]
detected_peaks_5000 = detected_peaks[detected_peaks < 5000]  # Limit peaks to the first 5000 values

# Plot the results
plt.figure(figsize=(10, 14))

# Original signal with detected peaks
plt.subplot(6, 1, 1)
plt.plot(ecg_matrix[:5000, 0], label='Original ECG Signal')
plt.plot(detected_peaks_5000, ecg_matrix[detected_peaks_5000, 0], "x", color='red', label='Detected Peaks')
plt.title('Original ECG Signal with Detected Peaks')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()

# Filtered signal
plt.subplot(6, 1, 2)
plt.plot(filtered_signal[:5000], color='orange')
plt.title(f'Filtered ECG Signal (Bandpass {Lowpass_setting}-{Highpass_setting} Hz)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# Differentiated signal
plt.subplot(6, 1, 3)
plt.plot(Differentiation_signal[:5000], color='green')
plt.title('Differentiated ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# Squared signal
plt.subplot(6, 1, 4)
plt.plot(Squared_signal[:5000], color='red')
plt.title('Squared ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# Moving average signal
plt.subplot(6, 1, 5)
plt.plot(Moving_average_signal[:5000], color='blue')
plt.title('Moving Average ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# Thresholded signal with detected peaks (limited to first 5000 samples)
plt.subplot(6, 1, 6)
plt.plot(Thresholded_signal_5000, color='purple', label='Thresholded Signal')
plt.plot(detected_peaks_5000, Thresholded_signal_5000[detected_peaks_5000], "x", color='black', label='Detected Peaks')
plt.axhline(y=Threshold, color='black', linestyle='--', label=f'Threshold = {Threshold:.2f}')
plt.title('Thresholded ECG Signal with Detected Peaks (First 5000 Samples)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()