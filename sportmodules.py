import os
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
from typing import Dict




def read_signal_raw(name: str, source: str) -> pd.DataFrame:

    POLAR_PATH = Path('../Data/08032023_radom/Polar')
    PNEUM_PATH = Path('../Data/08032023_radom/Pneumonitor')

    if source == 'polar':
        path = POLAR_PATH / f"{name}.csv"
        signal = pd.read_csv(path)

    elif source == 'pneum':
        path = PNEUM_PATH / f"{name}.DAT"
        signal = pd.read_csv(path, sep=';', header=None)

    else:
        print("Source not recognized. Use 'polar' or 'pneum'.")
        return None

    try:
        return signal
    except FileNotFoundError:
        print(f"File {path} not found.")
    except pd.errors.EmptyDataError:
        print(f"No data found in file {path}.")
    except Exception as e:
        print(f"An error occurred: {e}")




def preprocess_polar_signal(signal: np.ndarray, config: Dict[str, float]) -> np.ndarray:
    """
    Preprocesses the input signal by applying a series of filters based on the provided configuration.

    Parameters:
    signal (np.ndarray): The input signal to be processed.
    config (Dict[str, float]): Configuration dictionary containing filter parameters.

    Returns:
    np.ndarray: The filtered signal.
    """
    
    # Extract filter parameters from the configuration
    fs = config['fs']
    order = config['order']
    order_low = config['order_low']
    cutoff_high = config['cutoff_high']
    cutoff_low = config['cutoff_low']
    nyq = 0.5 * fs

    # Calculate normalized cutoff frequencies
    normal_cutoff_high = cutoff_high / nyq
    normal_cutoff_low = cutoff_low / nyq

    # Apply low-pass filter
    b, a = butter(order, normal_cutoff_high, btype="low")
    pre_filtered_signal = filtfilt(b, a, signal)

    # Apply high-pass filter
    b, a = butter(order_low, normal_cutoff_low, btype="high")
    filtered_signal = filtfilt(b, a, pre_filtered_signal)

    # Additional low-pass filter with new parameters
    order_extra = round((2 * fs) / 25)
    cutoff_extra = 35.81
    normal_cutoff_extra = cutoff_extra / nyq
    b, a = butter(order_extra, normal_cutoff_extra, btype="lowpass", analog=False)
    filtered_signal = filtfilt(b, a, filtered_signal)
    
    return filtered_signal

def get_cpet_time(cpet_signal):
    return cpet_signal.iloc[-1,0]


def get_cpet_lens(good_polar_recovery_ids):
    files = pd.read_csv('files.csv', sep = '\t')
    i = 0
    cpet_lens = []
    while i < len(good_polar_recovery_ids):
        cpet_id = files[files['polar_recovery'] == good_polar_recovery_ids[i]]['pneum_cpet'].values[0]
        cpet_signal = read_signal_raw(cpet_id, 'pneum')
        cpet_lens.append(get_cpet_time(cpet_signal))
        i+=1
        
    return cpet_lens


def get_RR_peaks(signal):
    peaks, _ = find_peaks(signal, height=500, distance=20)
    return peaks

def plot_ppg_RR(signal, RR, signal_label='PPG Signal', rr_label='R-R Peaks', title='PPG Signal Processed with RR'):
    fig, ax = plt.subplots()
    
    ax.plot(signal, label=signal_label)
    ax.plot(RR, signal[RR], "x", label=rr_label, color='red')
    ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    
    plt.show()
    
def get_HR_from_RR(RR, config):
    """
    Calculate heart rate (HR) from R-R intervals.

    Parameters:
    RR (array-like): Array of R-R interval indices.
    config (dict): Configuration dictionary containing 'fs' (sampling frequency).

    Returns:
    np.ndarray: Array of heart rate values in beats per minute (bpm).
    """
    if len(RR) < 2:
        raise ValueError("RR interval array must contain at least two elements.")
    
    rr_intervals = np.diff(RR)  # Calculate differences between consecutive R-R intervals
    sampling_frequency = config['fs']
    
    # Calculate heart rate in beats per minute (bpm)
    hr_seconds = rr_intervals / sampling_frequency
    heart_rate = 60 / hr_seconds
    
    return heart_rate

def calculate_hrv_from_rest(RR):
    """
    Calculate HRV metrics (SDNN and RMSSD) from R-R intervals.

    Parameters:
    RR (array-like): Array of R-R interval indices.

    Returns:
    dict: Dictionary containing HRV metrics 'SDNN' and 'RMSSD'.
    """
    if len(RR) < 2:
        raise ValueError("RR interval array must contain at least two elements.")
    
    # Calculate successive R-R intervals
    rr_intervals = np.diff(RR)
    
    # Calculate SDNN (Standard Deviation of NN intervals)
    sdnn = np.std(rr_intervals, ddof=1)  # ddof=1 for sample standard deviation
    
    # Calculate RMSSD (Root Mean Square of Successive Differences)
    successive_differences = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_differences ** 2))
    
    return {'SDNN': sdnn, 'RMSSD': rmssd}


