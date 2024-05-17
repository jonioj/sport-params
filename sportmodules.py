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