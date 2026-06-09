import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import scipy.signal as signal
import random
from sklearn.decomposition import FastICA
import pywt
import matplotlib.pyplot as plt
import io
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


#Constants
fs = 200


# Load train.csv
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
# print(train_df.head())


def butter_bandpass(lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')    

# def apply_filter(data, lowcut, highcut, order=4):
#     b, a = butter_bandpass(lowcut, highcut, order)
#     return signal.filtfilt(b, a, data)


def apply_filter(eeg_df):
    b, a = butter_bandpass(lowcut=.5, highcut=40.0, order=4)
    for col in eeg_df.columns:
        # eeg_df[col] = apply_filter(eeg_df[col], 0.5, 40, 4)
        eeg_df[col] = signal.filtfilt(b, a, eeg_df[col])
    return eeg_df


montage_pairs = [
    # Parasagittal chains (left hemisphere)
    ("Fp1", "F3"), 
    ("F3", "C3"), 
    ("C3", "P3"), 
    ("P3", "O1"),
    
    # Parasagittal chains (right hemisphere)
    ("Fp2", "F4"), 
    ("F4", "C4"), 
    ("C4", "P4"), 
    ("P4", "O2"),
    
    # Central chain
    ("Fz", "Cz"), 
    ("Cz", "Pz"),
    
    # Temporal chains (left side: Fp1 → F7 → T3 → T5 → O1)
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    
    # Temporal chains (right side: Fp2 → F8 → T4 → T6 → O2)
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2")
]

def apply_double_banana_montage(eeg_df: pd.DataFrame):
    montage_df = pd.DataFrame()
    for ch1, ch2 in montage_pairs:
        montage_df[f"{ch1}-{ch2}"] = eeg_df[ch1] - eeg_df[ch2]
    return montage_df


# def apply_ica(eeg_df: pd.DataFrame):    
#     ica = FastICA(n_components=eeg_df.shape[1], random_state=42, max_iter=1000)
#     try:
#         components = ica.fit_transform(eeg_df)
#         cleaned = ica.inverse_transform(components)
#         return pd.DataFrame(cleaned, columns=eeg_df.columns)
#     except Exception as e:
#         print(f"ICA failed: {e}")
#         return eeg_df


def crop_first_segment(eeg_df, duration_sec=50):
    num_samples = duration_sec * fs
    #start = (len(eeg_df) - num_samples) // 2
    start = 0 #take first 50 sec
    return eeg_df.iloc[start:start+num_samples].reset_index(drop=True)


def compute_power_matrix(signal, scales=np.arange(1, 64), wavelet='morl'):
    coefficients, _ = pywt.cwt(signal, scales, wavelet)
    power = (np.abs(coefficients)) ** 2
    return power

def compute_group_spectrogram(montage_df, group_channels, scales=np.arange(1, 64), wavelet='morl'):
    power_matrices = []
    for ch in group_channels:
        if ch in montage_df.columns:
            signal = montage_df[ch].values
            power = compute_power_matrix(signal, scales, wavelet)
            power_matrices.append(power)
    if not power_matrices:
        return None
    # Average pixel-wise over the computed power matrices
    avg_power = np.mean(np.array(power_matrices), axis=0)
    return avg_power


def display_power_matrix_as_image(power_matrix, signal_length, fs, scales, wavelet, title="", figsize=(6,4)):    
    # Total time span in seconds
    T = signal_length / fs
    
    freq_axis = pywt.scale2frequency(wavelet, scales) * fs
    
    extent = [0, T, freq_axis[-1], freq_axis[0]]

    plt.figure(figsize=figsize)
    plt.imshow(power_matrix, extent=extent, cmap='jet', aspect='auto', origin='lower')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar()
    plt.show()


chain_groups = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],  # Left Temporal Chain
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],  # Right Temporal Chain
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],  # Left Parasagittal Chain
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]   # Right Parasagittal Chain
}


def normalize_eeg(df: pd.DataFrame):
    return (df - df.mean()) / df.std()


def plot_eeg(df, title=""):
    n_samples_5sec = 25 * fs
    
    # df = montaged_ica
    
    eeg_df_first_5sec = df.iloc[:n_samples_5sec]
    
    # Plot settings
    plt.figure(figsize=(15, 8))
    n_channels = len(df.columns) 
    y_offsets = [i * 300 for i in range(n_channels)]
    
    # Plot each EEG channel
    for i, col in enumerate(df.columns[:n_channels]):
        plt.plot(
            eeg_df_first_5sec.index / fs,
            eeg_df_first_5sec[col] + y_offsets[i],
            linewidth=1,
        )
    
    plt.yticks(y_offsets, df.columns[:n_channels])
    plt.xlabel("Time (seconds)")
    plt.title(f"EEG {eeg_id} - {title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # plt.savefig("eeg_plot_after_ica.png", dpi=300)


eeg_id = 1002379034 #1000913311 #582999 #568657 #778705 #751790 #568657 #582999
eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{eeg_id}.parquet"
eeg_df = pd.read_parquet(eeg_path)

eeg_df = eeg_df.drop(columns=['EKG'])  # Skip EKG

plot_eeg(eeg_df, "raw eeg")

filtered_eeg = apply_filter(eeg_df)

plot_eeg(filtered_eeg, "after butterworth band filtering")

montaged_df = apply_double_banana_montage(filtered_eeg)

plot_eeg(montaged_df, "after applying double banana montage")

# montaged_ica = apply_ica(montaged_df)

# plot_eeg(montaged_ica, "after applying ica")

# middle_segment = crop_first_segment(montaged_ica)
first_segment = crop_first_segment(montaged_df)

# normalized_segment = normalize_eeg(first_segment)

# plot_eeg(normalized_segment, "after normalization")




scales = np.arange(1, 64)
wavelet = 'morl'

spectrogram_results = {}

for group_name, channels in chain_groups.items():
    power_matrix = compute_group_spectrogram(normalized_segment, channels, scales=scales, wavelet=wavelet)
    if power_matrix is not None:
        spectrogram_results[group_name] = power_matrix
        display_power_matrix_as_image(
            power_matrix,
            signal_length=first_segment.shape[0],
            fs=fs,
            scales=scales,
            wavelet=wavelet,
            title=f"{group_name} Spectrogram"
        )
    else:
        print(f"No valid channels found for group {group_name}.")


for group_name, channels in chain_groups.items():
    power_matrix = compute_group_spectrogram(first_segment, channels, scales=scales, wavelet=wavelet)
    if power_matrix is not None:
        spectrogram_results[group_name] = power_matrix
        display_power_matrix_as_image(
            power_matrix,
            signal_length=first_segment.shape[0],
            fs=fs,
            scales=scales,
            wavelet=wavelet,
            title=f"{group_name} Spectrogram"
        )
    else:
        print(f"No valid channels found for group {group_name}.")


# df = montaged_df

# eeg_df_first_5sec = df.iloc[:n_samples_5sec]

# # Plot settings
# plt.figure(figsize=(15, 8))
# n_channels = len(df.columns) 
# y_offsets = [i * 300 for i in range(n_channels)]

# # Plot each EEG channel
# for i, col in enumerate(df.columns[:n_channels]):
#     plt.plot(
#         eeg_df_first_5sec.index / sampling_rate,
#         eeg_df_first_5sec[col] + y_offsets[i],
#         linewidth=1,
#     )

# plt.yticks(y_offsets, df.columns[:n_channels])
# plt.xlabel("Time (seconds)")
# plt.title(f"EEG {eeg_id} - First 5 Seconds")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

