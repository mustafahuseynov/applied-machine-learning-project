import numpy as np
import pandas as pd
from scipy.signal import detrend, windows


def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with the fast fourier transformation.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequencies for each signal, shape: (num_samples,).
    """
    # Remove the mean/trend from each signal
    x_detrended = detrend(x, axis=1)

    # Apply a Hann window to reduce spectral leakage
    window = windows.hann(x.shape[1])
    x_windowed = x_detrended * window

    # Compute the FFT for each signal
    fft_result = np.fft.rfft(x_windowed, axis=1)

    # Calculate the Power Spectral Density (PSD)
    psd = np.abs(fft_result) ** 2

    # Generate the corresponding frequency values
    frequencies = np.fft.rfftfreq(x.shape[1], d=1 / fs)

    # Find the frequency with the highest PSD for each signal
    dominant_indices = np.argmax(psd, axis=1)
    dominant_frequencies = frequencies[dominant_indices]

    return dominant_frequencies


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extract 20 different features from the data.
    Args:
        data (np.ndarray): The data to extract features from.
        labels (np.ndarray): The labels of the data.
    Returns:
        pd.DataFrame: The extracted features.
    """
    fs = 100_000  # Sampling frequency (adapt if specified otherwise)

    features = {}

    # Separate voltage and current signals
    voltage = data[:, :, 0]
    current = data[:, :, 1]

    # Statistical features: voltage
    features["voltage_mean"] = np.mean(voltage, axis=1)
    features["voltage_std"] = np.std(voltage, axis=1)
    features["voltage_median"] = np.median(voltage, axis=1)
    features["voltage_min"] = np.min(voltage, axis=1)
    features["voltage_max"] = np.max(voltage, axis=1)
    features["voltage_rms"] = np.sqrt(np.mean(voltage ** 2, axis=1))

    # Frequency features: voltage
    features["voltage_dominant_frequency"] = find_dominant_frequencies(
        voltage, fs
    )

    voltage_fft = np.fft.rfft(voltage, axis=1)
    features["voltage_spectral_power"] = np.sum(
        np.abs(voltage_fft) ** 2, axis=1
    )

    # Statistical features: current
    features["current_mean"] = np.mean(current, axis=1)
    features["current_std"] = np.std(current, axis=1)
    features["current_median"] = np.median(current, axis=1)
    features["current_min"] = np.min(current, axis=1)
    features["current_max"] = np.max(current, axis=1)
    features["current_rms"] = np.sqrt(np.mean(current ** 2, axis=1))

    # Frequency features: current
    features["current_dominant_frequency"] = find_dominant_frequencies(
        current, fs
    )

    current_fft = np.fft.rfft(current, axis=1)
    features["current_spectral_power"] = np.sum(
        np.abs(current_fft) ** 2, axis=1
    )

    # Power-related features
    instantaneous_power = voltage * current

    features["mean_power"] = np.mean(instantaneous_power, axis=1)
    features["power_std"] = np.std(instantaneous_power, axis=1)

    features["apparent_power"] = (
        features["voltage_rms"] * features["current_rms"]
    )

    features["mean_absolute_power"] = np.mean(
        np.abs(instantaneous_power), axis=1
    )

    # Create DataFrame
    df = pd.DataFrame(features)

    # Add quality labels
    df["quality"] = labels

    return df

