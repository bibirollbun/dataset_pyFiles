# unzip train and test

import zipfile
import os

zip_file_path = '/kaggle/input/inria-bci-challenge/train.zip'
extract_to = '/kaggle/working/train'

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"File extracted to {extract_to}")


zip_file_path = '/kaggle/input/inria-bci-challenge/test.zip'
extract_to = '/kaggle/working/test'

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"File extracted to {extract_to}")


import pandas as pd
import os
import glob
import mne
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, List


print(os.listdir('/kaggle/working/train'))


df = pd.read_csv('/kaggle/working/train/Data_S02_Sess01.csv')
print(df.columns)
df['FeedBackEvent'].value_counts()


#shows the feedbacks which are equal to 1 using red dotted lines. the graph is of the
#Cz channel.
plt.figure(figsize=(14, 5))
plt.plot(df['Time'], df['Pz'], label='Pz (brain channel)')
feedback_onsets = df[df['FeedBackEvent'] == 1]['Time']  
for t in feedback_onsets:
    plt.axvline(x=t, color='r', linestyle='--', alpha=0.6)
plt.xlabel('Time (ms or sample)')
plt.ylabel('EEG amplitude')
plt.title('EEG - Cz channel with feedback onsets')
plt.legend()
plt.show()
num_lines = len(feedback_onsets)
print("Number of red lines (feedback events):", num_lines)


import os
import pandas as pd

def create_mapping(output_csv):  
    mapping_rows = []
    
    # Get list of EEG files
    eeg_dir = '/kaggle/working/train'
    eeg_files = [f for f in os.listdir(eeg_dir)]

    preds_df = pd.read_csv('/kaggle/input/inria-bci-challenge/TrainLabels.csv')
    pred_dict = dict(zip(preds_df["IdFeedBack"], preds_df["Prediction"]))

    for eeg_filename in eeg_files:
        eeg_path = os.path.join(eeg_dir, eeg_filename)
        eeg = pd.read_csv(eeg_path)

        # Remove Data word from the filename
        subj_sess = os.path.splitext(eeg_filename)[0].replace('Data_', '')  # "S02_sess01"

        # Extract rows where feedback is 1
        fb_events = eeg[eeg['FeedBackEvent'] == 1].reset_index(drop=True)

        # Label feedback values according to the time they occured 
        for idx, row in fb_events.iterrows():
            event_num = idx + 1
            fb_id = f"{subj_sess}_FB{str(event_num).zfill(3)}"
            
            mapping_rows.append({
                'EEG_File': eeg_filename,
                'Feedback_ID': fb_id,
                'Time': row['Time'],
                'Prediction': pred_dict.get(fb_id, None)
            })
    
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(output_csv, index=False)
    
    print(f"Mapping created with {len(mapping_df)} rows. Saved to {output_csv}")
    return mapping_df
    
mapping_df = create_mapping(
    output_csv="feedback_mapping.csv"
)

print("mapping_df.csv created successfully!")
print(mapping_df.head())


import pandas as pd
import numpy as np
from scipy.signal import iirnotch, filtfilt
from typing import List

def apply_notch_filter(df,eeg_channels: List[str], f0=50.0, Q=60.0, fs=200):
    df_filtered = df.copy()
    b, a = iirnotch(f0, Q, fs)
    # Filter only the EEG channels, leaving other columns (like EOG) untouched
    for channel in eeg_channels:
        if channel in df_filtered.columns:
            df_filtered[channel] = filtfilt(b, a, df_filtered[channel])
            
    return df_filtered

# df = pd.read_csv('/kaggle/working/train/Data_S02_Sess01.csv')
# filtered_eeg_channels_df = apply_notch_filter(df)
# # Apply the function to your DataFrame
# filtered_df = pd.concat([df[['Time', 'FeedBackEvent','EOG']], filtered_eeg_channels_df], axis=1)

# plot_samples = 400 # Plot the first 2 seconds of data (400 samples / 200 Hz)
# plt.figure(figsize=(15, 6))
# plt.plot(df['Time'][:plot_samples], df['Cz'][:plot_samples], label='Original Cz Signal', color='red', alpha=1)
# plt.plot(filtered_df['Time'][:plot_samples], filtered_df['Cz'][:plot_samples], label='Filtered Cz Signal (50Hz notch)', color='blue', linewidth=1)
# plt.xlabel('Time (s)')
# plt.ylabel('EEG Amplitude (µV)')
# plt.title('Comparison of Original vs. Notch-Filtered EEG Signal (Cz Channel)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# import matplotlib.pyplot as plt
# from scipy.signal import welch

# # Parameters
# fs = 200  # Sampling frequency
# nperseg = 1024  # segment length for Welch (controls smoothness)

# # Get Cz channel signals
# original_cz = df['Cz'].values
# filtered_cz = filtered_df['Cz'].values

# # Compute power spectral density (PSD) using Welch
# f_orig, Pxx_orig = welch(original_cz, fs, nperseg=nperseg)
# f_filt, Pxx_filt = welch(filtered_cz, fs, nperseg=nperseg)

# # Plot frequency domain comparison
# plt.figure(figsize=(15, 6))
# plt.semilogy(f_orig, Pxx_orig, label='Original Cz Signal', color='red', alpha=0.8)
# plt.semilogy(f_filt, Pxx_filt, label='Filtered Cz Signal (50Hz notch)', color='blue', alpha=0.8)

# plt.axvline(50, color='k', linestyle='--', label='50 Hz (Notch)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (µV²/Hz)')
# plt.title('Frequency Domain (Cz Channel) - Before vs After Notch Filtering')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlim(0, 100)  # EEG is usually meaningful up to Nyquist (fs/2 = 100 Hz)
# plt.show()



#reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC6956025/
#for removal of atifacts i've used a combination of linear regression model and ica 
# 1) linear regression is applied to remove linear artifacts
# 2) seperating into independent components
# 3) eog corelation from the given eog data
# 4) wavelet enhancement, in the form of component correciton rather than component rejection
# 5) reconstruction
import mne
import numpy as np
import pywt
import pandas as pd
from scipy.signal import find_peaks # Import find_peaks
import matplotlib.pyplot as plt

def clean_eeg_pipeline_corrected(
    df: pd.DataFrame,
    sfreq: int ,
    eeg_channels: list,
    eog_channels: list,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    n_components: int = 56,
    random_state: int = 97,
    wavelet: str = 'db4',
    wavelet_level: int = 4,
    wavelet_thresh_factor: float = 3.0,
    peak_thresh_sd: float = 2.5, # Threshold in std devs for peak detection
    peak_distance_s: float = 0.5 # Min distance between peaks in seconds
) -> dict:
    """
    peak_thresh_sd (float, optional): Standard deviation multiplier to set the
                                     threshold for detecting artifact peaks.
    peak_distance_s (float, optional): Minimum required distance between detected
                                       peaks in seconds.
    """
    # Steps 0-2
    ch_names = eeg_channels + eog_channels
    ch_types = ['eeg'] * len(eeg_channels) + ['eog'] * len(eog_channels)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(df[ch_names].T.values * 1e-6, info, verbose='ERROR')


    raw_unfiltered = raw.copy()
    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge')

    eeg_data = raw.get_data(picks=eeg_channels)
    eog_data = raw.get_data(picks=eog_channels)
    eeg_cleaned_reg = np.zeros_like(eeg_data)

    #apply linear regression
    for i in range(eeg_data.shape[0]):
        beta = np.linalg.lstsq(eog_data.T, eeg_data[i, :], rcond=None)[0]
        eeg_cleaned_reg[i, :] = eeg_data[i, :] - eog_data.T @ beta
    raw_reg_cleaned = raw.copy()
    raw_reg_cleaned._data[:len(eeg_channels), :] = eeg_cleaned_reg

    # Step 3: ICA Decomposition
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    ica.fit(raw_reg_cleaned)

    # Step 4: Identify EOG components to be CORRECTED
    eog_indices_to_correct, eog_scores = ica.find_bads_eog(raw_reg_cleaned, ch_name=eog_channels)

    # Step 5: Wavelet Component Correction 
    print(f"Found {len(eog_indices_to_correct)} EOG-related components to correct: {eog_indices_to_correct}")

    # Get the original source signals
    ica_sources = ica.get_sources(raw_reg_cleaned).get_data()
    # Create a copy that we will modify
    corrected_sources = ica_sources.copy()

    def wavelet_denoise_segment(segment):
        coeffs = pywt.wavedec(segment, wavelet, level=wavelet_level)
        for i in range(1, len(coeffs)):
            sigma = np.median(np.abs(coeffs[i])) / 0.6745
            threshold = wavelet_thresh_factor * sigma
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        return pywt.waverec(coeffs, wavelet)

    # Loop over only the components identified as EOG-related
    for ic_index in eog_indices_to_correct:
        ic_signal = ica_sources[ic_index, :]
        
        # 1. Detect peaks (blinks/saccades) in the component
        peak_height_thresh = np.std(ic_signal) * peak_thresh_sd
        min_peak_dist = int(sfreq * peak_distance_s)
        peaks, _ = find_peaks(np.abs(ic_signal), height=peak_height_thresh, distance=min_peak_dist)
        
        if len(peaks) == 0:
            continue # No artifacts found, move to the next component

        # 2. Define 1-second windows around each peak
        window_radius = sfreq // 2
        artifact_windows = [(max(0, p - window_radius), min(len(ic_signal), p + window_radius)) for p in peaks]

        # 3. Merge overlapping windows to create continuous artifact segments
        if not artifact_windows:
            continue
        
        # Sort windows by start time
        artifact_windows.sort(key=lambda interval: interval[0])
        merged = [artifact_windows[0]]
        for current in artifact_windows[1:]:
            prev = merged[-1]
            if current[0] <= prev[1]: # Overlap
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        
        # 4. Apply wavelet denoising only within the merged artifact segments
        for start, end in merged:
            original_segment = ic_signal[start:end]
            corrected_segment = wavelet_denoise_segment(original_segment)
            corrected_segment = corrected_segment[:len(original_segment)]
            # Place the corrected segment back into our sources array
            corrected_sources[ic_index, start:end] = corrected_segment

     # Step 6: Reconstruct Clean EEG
    cleaned_eeg_data = np.dot(ica.mixing_matrix_[:, :ica.n_components_], corrected_sources)
    
    # Create a new Raw object for the cleaned data
    info_eeg_only = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw_after = mne.io.RawArray(cleaned_eeg_data, info_eeg_only)

    return {
        'cleaned_data': cleaned_eeg_data,
        # 'raw_before': raw_reg_cleaned, # Data after filtering but before ICA correction
        # 'raw_after': raw_after,
        # 'ica': ica,
        # 'eog_indices': eog_indices_to_correct
    }

# df = pd.read_csv('/kaggle/working/train/Data_S02_Sess01.csv')
# eeg_channels = [col for col in df.columns if col not in ['Time', 'FeedBackEvent','EOG']]
# eog_channels = ['EOG'] 
# print(f"Data loaded successfully.")
# print(f"Found {len(eeg_channels)} EEG channels.")
# sfreq=200
# results = clean_eeg_pipeline_corrected(
#     df=df,
#     sfreq = sfreq,
#     eeg_channels=eeg_channels,
#     eog_channels=eog_channels,
#     n_components=len(eeg_channels) 
# )

# # --- Configuration for the plot ---
# channel_to_plot_idx = 0  # Plot the first EEG channel
# channel_name = eeg_channels[channel_to_plot_idx] # Get its name
# duration_to_plot_s = 5  # seconds
# start_time_s = 10      # Start plotting from 10 seconds into the data

# # Calculate sample indices for the plotting duration
# start_sample = int(start_time_s * sfreq)
# end_sample = int((start_time_s + duration_to_plot_s) * sfreq)

# # Get the data for the selected channel
# # Ensure raw_before and raw_after objects exist in results
# if 'raw_before' in results and 'raw_after' in results:
#     # Extract data for the specific channel
#     original_eeg = results['raw_before'].get_data(picks=channel_name)[0, start_sample:end_sample] * 1e6 # convert to uV
#     cleaned_eeg = results['raw_after'].get_data(picks=channel_name)[0, start_sample:end_sample] * 1e6 # convert to uV

#     # Create a time vector for plotting
#     time = np.arange(0, len(original_eeg)) / sfreq

#     # --- Plotting ---
#     plt.figure(figsize=(12, 6))
#     plt.plot(time, original_eeg, color='red', linewidth=0.7, label=f'Original {channel_name} Signal (Before Cleaning)')
#     plt.plot(time, cleaned_eeg, color='blue', linewidth=0.7, label=f'Cleaned {channel_name} Signal (After Cleaning)')

#     plt.title(f'Comparison of {channel_name} Channel Before vs. After EOG Cleaning')
#     plt.xlabel('Time (s)')
#     plt.ylabel('EEG Amplitude (µV)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Error: 'raw_before' or 'raw_after' not found in results. Please ensure the pipeline returns these.")

# print("Plotting the power spectrum before and after cleaning...")
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
# results['raw_before'].plot_psd(ax=ax1, show=False, fmax=50)
# ax1.set_title('PSD Before Cleaning')
# ax1.set_xlabel('Frequency (Hz)')
# results['raw_after'].plot_psd(ax=ax2, show=False, fmax=50)
# ax2.set_title('PSD After Cleaning')
# ax2.set_xlabel('Frequency (Hz)')
# fig.suptitle('Power Spectral Density Comparison')
# plt.tight_layout()
# plt.show()




def create_epochs_with_rejection(
    eeg_csv_path: str, feedback_df_group: pd.DataFrame, sfreq: float,
    all_eeg_chans: List[str], feature_chans: List[str],
    tmin: float, tmax: float, reject_threshold_uv: float = 150.0
) -> Optional[mne.Epochs]:
    """
    A simpler pipeline that loads data, applies standard filters, and uses
    peak-to-peak amplitude rejection to remove major artifacts.
    """
    print(f"[INFO] Reading continuous data from: {os.path.basename(eeg_csv_path)}")
    try:
        eeg_df = pd.read_csv(eeg_csv_path, usecols=all_eeg_chans)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Could not read or find required channels in {eeg_csv_path}: {e}")
        return None

    # 1. Create MNE Raw object, assuming input is in µV and converting to V
    info = mne.create_info(ch_names=all_eeg_chans, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_df[all_eeg_chans].T.values * 1e-6, info, verbose='ERROR')
    
    # 2. Apply a standard band-pass filter to remove slow drifts and high-frequency noise
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose='ERROR')
    
    # 3. Select only the channels we need for feature extraction
    raw.pick_channels(feature_chans)
    
    # 4. Create an MNE events array from feedback timestamps
    event_samples = (feedback_df_group['Time'].values * sfreq).astype(int)
    events = np.vstack([event_samples, np.zeros_like(event_samples), np.ones_like(event_samples)]).T

    # 5. Create the Epochs object WITH ARTIFACT REJECTION
    # This is the key step. It will automatically drop any epoch where the
    # signal jumps by more than the rejection threshold.
    reject_criteria = dict(eeg=reject_threshold_uv * 1e-6) # Convert µV to V for MNE

    try:
        epochs = mne.Epochs(
            raw, events=events, tmin=tmin, tmax=tmax,
            baseline=(None, 0),
            reject=reject_criteria, # <-- THE NEW PARAMETER
            preload=True,
            verbose='ERROR'
        )
        
        # We need to know which events were kept to align metadata
        n_events_original = len(events)
        n_epochs_created = len(epochs)
        if n_events_original != n_epochs_created:
            print(f"[INFO] Dropped {n_events_original - n_epochs_created} epochs due to artifacts or boundary issues.")

        metadata_kept = feedback_df_group.iloc[epochs.selection].reset_index(drop=True)
        epochs.metadata = metadata_kept
        
        return epochs
    except ValueError as e:
        print(f"[WARN] Could not create epochs for {os.path.basename(eeg_csv_path)}. Error: {e}")
        return None


# EEG feature extraction 
# https://pmc.ncbi.nlm.nih.gov/articles/PMC1868547/
# the relevant extraction is done using ERP (Event-Related Potentials)
# 1. average the signal in short time windows after feedback
# (e.g., 200–300 ms for FRN (Feedback Related Negativity) at FCz/Cz, 300–600 ms for P300 (a positive peak around 300 ms after an infrequent target appears) at Pz).
# 2. compute 4–8 Hz power ~200–400 ms at FCz 
# the brain also produces short burst of specific rhythm (a "theta wave") when processing feedback.

from __future__ import annotations
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert, iirnotch, filtfilt, find_peaks
import pywt # You may need to run: pip install PyWavelets

# ==============================================================================
# CONFIG — EDIT THESE VALUES
# ==============================================================================
# --- Input Paths ---
FEEDBACK_CSV_PATH = "/kaggle/working/feedback_mapping.csv"
EEG_DATA_DIR      = "/kaggle/working/train"

# --- File & Column Naming ---
EEG_FILE_EXTENSION = ".csv"
FILE_COL           = "EEG_File"
TIMESTAMP_COL      = "Time"
LABEL_COL          = "prediction"
# **NEW**: Define both EEG and EOG channel names from your CSV files
EEG_CHANNEL_COLS   = ['FCz', 'Cz', 'Pz'] # Channels to use for feature extractionA
EOG_CHANNEL_COLS   = ['EOG'] # The name of your EOG channel column
# OPTIONAL_META      = ["subject_id", "dataset_id", "session_id", "trial_id"]

# --- Preprocessing & Epoching Parameters ---
SAMPLING_RATE_HZ = 200.0  # The sampling frequency of your EEG data
EPOCH_TMIN       = -0.2   # Start time of an epoch in seconds
EPOCH_TMAX       = 1.0    # End time of an epoch in seconds

# --- Output Files ---
OUTPUT_CSV      = "all_training_data_cleaned.csv"
OUTPUT_PICKLE   = "all_training_data_cleaned.pkl"

# ==============================================================================
# UPDATED HELPER: Create Epochs from Cleaned CSV Data
# ==============================================================================
def create_epochs_from_csv(
    eeg_csv_path: str, feedback_df_group: pd.DataFrame, sfreq: float,
    all_eeg_chans: List[str], eog_chans: List[str], feature_chans: List[str],
    tmin: float, tmax: float
) -> Optional[mne.Epochs]:
    """
    Loads, cleans (notch + ICA), and epochs the EEG data from a CSV file.
    """
    print(f"[INFO] Reading continuous data from: {os.path.basename(eeg_csv_path)}")
    try:
        # 1. Load the raw data from the CSV file
        eeg_df = pd.read_csv(eeg_csv_path, usecols=all_eeg_chans + eog_chans)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Could not read or find required channels in {eeg_csv_path}: {e}")
        return None
    
    # 2. Apply the 50Hz notch filter
    print("Applying 50Hz notch filter...")
    eeg_df_notched = apply_notch_filter(eeg_df, eeg_channels=all_eeg_chans, fs=sfreq)

    # 3. Apply the advanced cleaning pipeline (ICA + Wavelet)
    print("Applying advanced EOG artifact correction pipeline...")
    cleaning_results = clean_eeg_pipeline_corrected(
        df=eeg_df_notched, sfreq=int(sfreq), eeg_channels=all_eeg_chans, eog_channels=eog_chans
    )
    cleaned_data = cleaning_results['cleaned_data'] # This data is now in Volts

    # 4. Create an MNE Info object and RawArray from the CLEANED data
    info = mne.create_info(ch_names=all_eeg_chans, sfreq=sfreq, ch_types='eeg')
    raw_cleaned = mne.io.RawArray(cleaned_data, info, verbose='ERROR')
    
    # 5. Select only the channels needed for feature extraction
    raw_cleaned.pick_channels(feature_chans)

    # 6. Create an MNE events array from feedback timestamps
    event_samples = (feedback_df_group[TIMESTAMP_COL].values * sfreq).astype(int)
    events = np.vstack([event_samples, np.zeros_like(event_samples), np.ones_like(event_samples)]).T

    # 7. Create the Epochs object
    try:
        epochs = mne.Epochs(
            raw_cleaned, events=events, tmin=tmin, tmax=tmax,
            baseline=(None, 0), preload=True, verbose='ERROR'
        )
    except ValueError as e:
        print(f"[WARN] Could not create epochs for {os.path.basename(eeg_csv_path)}. Error: {e}")
        return None

    epochs.metadata = feedback_df_group.copy().reset_index(drop=True)
    return epochs

def _time_to_samples(epochs: mne.Epochs, tmin_ms: float, tmax_ms: float) -> Tuple[int, int]:
    t = epochs.times
    i0 = np.searchsorted(t, tmin_ms / 1000.0, side='left')
    i1 = np.searchsorted(t, tmax_ms / 1000.0, side='right')
    return i0, i1

def _theta_power_hilbert(epochs: mne.Epochs, ch_name: str, tmin_ms: float, tmax_ms: float, fmin: float = 4., fmax: float = 8.) -> np.ndarray:
    ep_ch = epochs.copy().pick(ch_name).filter(fmin, fmax, method='fir', phase='zero', verbose='ERROR')
    data = ep_ch.get_data()[:, 0, :]
    analytic = hilbert(data, axis=-1)
    power = np.abs(analytic) ** 2
    i0, i1 = _time_to_samples(ep_ch, tmin_ms, tmax_ms)
    return power[:, i0:i1].mean(axis=1) if i1 > i0 else np.full((power.shape[0],), np.nan)

def extract_bci_features(epochs: mne.Epochs, to_microvolts: bool = True) -> pd.DataFrame:
    frn_win, p300_win, theta_win = (200, 300), (300, 600), (200, 400)
    missing = [ch for ch in EEG_CHANNEL_COLS if ch not in epochs.ch_names]
    if missing: raise ValueError(f"Missing required feature channels: {missing}.")

    X = epochs.get_data()
    if to_microvolts: X *= 1e6

    i0_frn, i1_frn = _time_to_samples(epochs, *frn_win)
    i0_p3, i1_p3 = _time_to_samples(epochs, *p300_win)

    idx_FCz, idx_Cz, idx_Pz = [epochs.ch_names.index(ch) for ch in EEG_CHANNEL_COLS]

    frn_FCz = X[:, idx_FCz, i0_frn:i1_frn].mean(axis=1)
    frn_Cz = X[:, idx_Cz, i0_frn:i1_frn].mean(axis=1)
    p300_Pz = X[:, idx_Pz, i0_p3:i1_p3].mean(axis=1)
    theta_power_FCz = _theta_power_hilbert(epochs, 'FCz', *theta_win)

    return pd.DataFrame({
        'FRN_FCz_mean_200_300ms': frn_FCz,
        'FRN_Cz_mean_200_300ms': frn_Cz,
        'P300_Pz_mean_300_600ms': p300_Pz,
        'Theta_4_8Hz_power_FCz_200_400ms': theta_power_FCz,
    })

# ==============================================================================
# Main Build Function (Updated to call new epoch creation function)
# ==============================================================================
def build_training_table() -> pd.DataFrame:
    fb = pd.read_csv(FEEDBACK_CSV_PATH)
    fb["_file_key"] = fb[FILE_COL].str.replace(r'\.csv$', '', regex=True)
    all_rows = []
    for file_key, group in fb.groupby("_file_key"):
        print(f"\n---> Now processing file: {file_key}, which has {len(group)} events listed in the CSV.")
        # if file_key == 'Data_S02_Sess01':
        #     print("\n--- RUNNING DIAGNOSTICS for data_s02_sess01 ---")
            
        #     # 1. Get all timestamps for this file (in seconds)
        #     timestamps_s = group[TIMESTAMP_COL].values
            
        #     # 2. Check for NaN or missing timestamps
        #     nan_count = np.isnan(timestamps_s).sum()
        #     if nan_count > 0:
        #         print(f"[PROBLEM] Found {nan_count} NaN timestamps!")
    
        #     # 3. Define boundaries
        #     start_boundary = -EPOCH_TMIN  # This will be 0.2 seconds
        #     # To get the end boundary, we have to load the EEG data first
        #     eeg_csv_path = os.path.join(EEG_DATA_DIR, f"{file_key}{EEG_FILE_EXTENSION}")
        #     try:
        #         # We only need to load it to check the duration
        #         eeg_df = pd.read_csv(eeg_csv_path)
        #         eeg_duration_s = len(eeg_df) / SAMPLING_RATE_HZ
        #         end_boundary = eeg_duration_s - EPOCH_TMAX
        #         print(f"Verified EEG recording duration: {eeg_duration_s:.3f} seconds")
        #     except FileNotFoundError:
        #         print("[ERROR] Could not find the EEG file to verify duration.")
        #         continue # Skip to next file in the main loop
    
        #     # 4. Find and list the events that violate the boundaries
        #     early_events = timestamps_s[timestamps_s < start_boundary]
        #     late_events = timestamps_s[timestamps_s > end_boundary]
    
        #     print(f"\nChecking against start boundary ({start_boundary:.3f}s) and end boundary ({end_boundary:.3f}s)...")
    
        #     if len(early_events) > 0:
        #         print(f"[PROBLEM] Found {len(early_events)} events that are too close to the start:")
        #         print(np.round(early_events, 3))
            
        #     if len(late_events) > 0:
        #         print(f"[PROBLEM] Found {len(late_events)} events that are too close to the end:")
        #         print(np.round(late_events, 3))
                
        #     if len(early_events) == 0 and len(late_events) == 0 and nan_count == 0:
        #         print("[INFO] No boundary violations found for this file.")
    
        #     print("--- END DIAGNOSTICS ---\n")
            
        eeg_csv_path = os.path.join(EEG_DATA_DIR, f"{file_key}{EEG_FILE_EXTENSION}")
        if not os.path.exists(eeg_csv_path):
            print(f"[WARN] Could not find EEG CSV for '{file_key}'. Skipping.")
            continue
        else:
            df = pd.read_csv(eeg_csv_path)
        ALL_EEG_CHANNELS = [col for col in df.columns if col not in ['Time', 'FeedBackEvent', 'EOG']]
    
        # Call the updated function that now includes the cleaning steps
        epochs = create_epochs_with_rejection(
            eeg_csv_path, group, sfreq=SAMPLING_RATE_HZ,
            all_eeg_chans=ALL_EEG_CHANNELS,
            feature_chans=EEG_CHANNEL_COLS, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX
        )

        if epochs is None or len(epochs) == 0:
            print(f"[WARN] No valid epochs were created for '{file_key}'. Skipping.")
            continue

        feats = extract_bci_features(epochs)
        out = epochs.metadata.merge(feats, left_index=True, right_index=True)
        all_rows.append(out)

    if not all_rows:
        raise RuntimeError("No data could be processed. Check file paths and channel names.")

    all_data = pd.concat(all_rows, axis=0, ignore_index=True)
    
    feature_cols = [c for c in all_data.columns if c.startswith(("FRN_", "P300_", "Theta_"))]
    all_data.dropna(subset=feature_cols, inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    all_data.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Saved unified training table to: {OUTPUT_CSV} ({len(all_data)} rows)")
    all_data.to_pickle(OUTPUT_PICKLE)
    return all_data

# ==============================================================================
# Run the script
# ==============================================================================
if __name__ == "__main__":
    mne.set_log_level('WARNING')
    final_df = build_training_table()
    print("\n--- First 5 rows of the final cleaned dataset ---")
    print(final_df.head())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset you created
df = pd.read_csv("all_training_data_cleaned.csv")

# Pick a key feature, like the P300
feature_to_plot = 'P300_Pz_mean_300_600ms'
target_col = 'Prediction' # Or 'label' if you named it that

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x=feature_to_plot, hue=target_col, fill=True, common_norm=False)
plt.title(f'Distribution of {feature_to_plot} by Class')
plt.xlabel('Mean Amplitude (µV)')
plt.grid(True, linestyle='--')
plt.show()


import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

# ==============================================================================
# IMPORTANT: PASTE ALL YOUR HELPER FUNCTIONS HERE
# You need the full definitions of:
# - apply_notch_filter
# - clean_eeg_pipeline_corrected
# - create_epochs_from_csv
# - _time_to_samples
# - _theta_power_hilbert
# - extract_bci_features
# ==============================================================================
# (All helper functions from your main script would go here)


# ==============================================================================
# CONFIGURATION FOR VISUALIZATION
# ==============================================================================
# --- Choose the single file you want to analyze ---
FILE_TO_ANALYZE = "Data_S02_Sess01" # The file that had issues, for example

# --- Set paths and parameters (should match your main script) ---
FEEDBACK_CSV_PATH = "/kaggle/working/feedback_mapping.csv"
EEG_DATA_DIR      = "/kaggle/working/train"
EEG_FILE_EXTENSION = ".csv"
FILE_COL           = "EEG_File"
TIMESTAMP_COL      = "Time"
EEG_CHANNEL_COLS   = ['FCz', 'Cz', 'Pz']
EOG_CHANNEL_COLS   = ['EOG']
SAMPLING_RATE_HZ = 200.0
EPOCH_TMIN       = -0.2
EPOCH_TMAX       = 1.0


# ==============================================================================
# SCRIPT TO GENERATE AND PLOT THE ERPs
# ==============================================================================

# 1. LOAD THE MAPPING FILE AND GET DATA FOR OUR CHOSEN FILE
print("Loading feedback mapping file...")
fb = pd.read_csv(FEEDBACK_CSV_PATH)
fb["_file_key"] = fb[FILE_COL].str.replace(r'\.csv$', '', regex=True)
group_to_process = fb[fb["_file_key"] == FILE_TO_ANALYZE]

if group_to_process.empty:
    raise ValueError(f"Could not find file key '{FILE_TO_ANALYZE}' in the feedback CSV.")

# 2. RUN THE FULL PROCESSING PIPELINE TO CREATE THE EPOCHS OBJECT
print(f"\nProcessing {FILE_TO_ANALYZE} to create epochs...")
eeg_csv_path = os.path.join(EEG_DATA_DIR, f"{FILE_TO_ANALYZE}{EEG_FILE_EXTENSION}")

# Dynamically get all channel names from the file
df_cols = pd.read_csv(eeg_csv_path, nrows=1).columns
ALL_EEG_CHANNELS = [col for col in df_cols if col not in ['Time', 'FeedBackEvent', 'EOG']]

# This is the crucial step that was missing before
epochs = create_epochs_with_rejection(
    eeg_csv_path, group_to_process, sfreq=SAMPLING_RATE_HZ,
    all_eeg_chans=ALL_EEG_CHANNELS,
    feature_chans=EEG_CHANNEL_COLS, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX
)

if epochs is None:
    raise RuntimeError("Epoch creation failed for the selected file.")

print(f"Successfully created {len(epochs)} epochs.")

# 3. SEPARATE THE CREATED EPOCHS BY CLASS (0 vs 1)
# Note: Use the correct column name from your metadata, e.g., 'prediction'
epochs_class_0 = epochs[epochs.metadata['Prediction'] == 0]
epochs_class_1 = epochs[epochs.metadata['Prediction'] == 1]
print(f"Found {len(epochs_class_0)} epochs for class 0.")
print(f"Found {len(epochs_class_1)} epochs for class 1.")


# 4. AVERAGE THE EPOCHS FOR EACH CLASS TO CREATE "EVOKED" RESPONSES
evoked_0 = epochs_class_0.average()
evoked_1 = epochs_class_1.average()

# 5. PLOT THE COMPARISON
print("\nPlotting the average ERPs for each class...")
# We can plot multiple channels at once
channels_to_plot = ['FCz', 'Cz', 'Pz']

fig = mne.viz.plot_compare_evokeds(
    {'Class 0': evoked_0, 'Class 1': evoked_1},
    picks=channels_to_plot,
    title=f'Average ERP for Class 0 vs. Class 1 on file {FILE_TO_ANALYZE}'
)
# Make the plot bigger for better visibility
for figure in fig:
    figure.set_size_inches(12, 8)

plt.show()


import pandas as pd

# Load one of your raw EEG files
df = pd.read_csv('/kaggle/working/train/Data_S02_Sess01.csv')

# Check the statistics of a key channel
print(df['Pz'].describe())


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- NEW: Import LDA and SVM ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# --- CONFIGURATION ---
TRAINING_DATA_PATH = "all_training_data_cleaned.csv"
MODEL_SAVE_PATH = "eeg_classifier_model.joblib"
SCALER_SAVE_PATH = "feature_scaler.joblib"
TARGET_COL = "Prediction"

# 1. LOAD THE DATASET
print(f"Loading feature-engineered dataset from: {TRAINING_DATA_PATH}")
df = pd.read_csv(TRAINING_DATA_PATH)

# 2. DEFINE FEATURES (X) AND TARGET (y)
feature_cols = [col for col in df.columns if col.startswith(('FRN_', 'P300_', 'Theta_'))]
X = df[feature_cols]
y = df[TARGET_COL]

print(f"\nFound {len(feature_cols)} features: {feature_cols}")

# 3. SPLIT DATA INTO TRAINING AND VALIDATION SETS
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# 4. SCALE THE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. INITIALIZE AND TRAIN THE MODEL
print("\n--- Model Selection ---")

# --- Option A: Linear Discriminant Analysis (LDA) ---
# Fast, simple, and a very strong baseline for BCI.
print("Initializing LDA model...")
model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# --- Option B: Support Vector Machine (SVM) ---
# Powerful but can be much slower to train, especially with large datasets.
# To use SVM, comment out the LDA line above and uncomment the two lines below.
# print("Initializing SVM model...")
# model = SVC(kernel='rbf', probability=True, random_state=42) # probability=True is essential!


print(f"\nTraining the {type(model).__name__} model...")
model.fit(X_train_scaled, y_train)
print("Training complete.")

# 6. SAVE THE TRAINED MODEL AND SCALER
joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print(f"Scaler saved to: {SCALER_SAVE_PATH}")

# 7. EVALUATE ON THE VALIDATION SET (using probabilities)
print("\n--- Evaluating on Validation Set ---")
# Get the probability predictions for the positive class (class 1)
y_proba_val = model.predict_proba(X_val_scaled)[:, 1]

# Calculate the primary metric: AUC score
auc_score = roc_auc_score(y_val, y_proba_val)
print(f"Validation AUC Score: {auc_score:.4f}")

# For a more traditional view, calculate accuracy at a 0.5 threshold
y_pred_val_thresholded = (y_proba_val >= 0.5).astype(int)
accuracy = accuracy_score(y_val, y_pred_val_thresholded)
print(f"\nValidation Accuracy (at 0.5 threshold): {accuracy:.4f}")
print("\nClassification Report (at 0.5 threshold):")
print(classification_report(y_val, y_pred_val_thresholded))

# Display the confusion matrix
cm = confusion_matrix(y_val, y_pred_val_thresholded)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Validation Confusion Matrix (at 0.5 threshold)')
plt.show()


import os
import pandas as pd

def create_test_mapping(eeg_dir, output_csv):
    """
    Scans a directory of raw EEG CSVs (test set), finds all feedback event
    timestamps, and creates a mapping file (event manifest) without labels.
    """
    mapping_rows = []
    
    # Get list of EEG files from the specified test directory
    eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.csv')]
    print(f"Found {len(eeg_files)} EEG files in {eeg_dir}")

    for eeg_filename in eeg_files:
        eeg_path = os.path.join(eeg_dir, eeg_filename)
        eeg = pd.read_csv(eeg_path)

        # Remove Data word from the filename
        subj_sess = os.path.splitext(eeg_filename)[0].replace('Data_', '') # e.g., "S21_Sess01"

        # Extract rows where feedback is 1
        fb_events = eeg[eeg['FeedBackEvent'] == 1].reset_index(drop=True)

        # Create a unique ID for each feedback event for tracking
        for idx, row in fb_events.iterrows():
            event_num = idx + 1
            # This ID is still useful for your final submission file
            fb_id = f"{subj_sess}_FB{str(event_num).zfill(3)}"
            
            mapping_rows.append({
                'EEG_File': eeg_filename,
                'Feedback_ID': fb_id,
                'Time': row['Time'],
            })
    
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(output_csv, index=False)
    
    print(f"Test mapping created with {len(mapping_df)} rows. Saved to {output_csv}")
    return mapping_df

TEST_EEG_DIR = '/kaggle/working/test'
test_mapping_df = create_test_mapping(
    eeg_dir=TEST_EEG_DIR,
    output_csv="test_feedback_mapping.csv"
)

print("\nTest mapping created successfully!")
print(test_mapping_df.head())


from __future__ import annotations
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert, iirnotch, filtfilt, find_peaks
import pywt # You may need to run: pip install PyWavelets
import joblib

# ==============================================================================
# CONFIG — VERIFY THESE VALUES MATCH YOUR TRAINING SETUP
# ==============================================================================
# --- Input Paths ---
TEST_EEG_DIR = '/kaggle/working/test'
TEST_FEEDBACK_MAPPING_CSV = "test_feedback_mapping.csv" # The event manifest you just created
MODEL_PATH = "eeg_classifier_model.joblib" # Saved model from your training script
SCALER_PATH = "feature_scaler.joblib"     # Saved scaler from your training script

# --- File & Column Naming ---
EEG_FILE_EXTENSION = ".csv"
FILE_COL           = "EEG_File"       # Column in your mapping CSV that identifies the EEG file
TIMESTAMP_COL      = "Time"           # Column in your mapping CSV for timestamps (in seconds)
# NOTE: The feature extraction channels MUST match what the model was trained on
EEG_CHANNEL_COLS   = ['FCz', 'Cz', 'Pz']
EOG_CHANNEL_COLS   = ['EOG']

# --- Preprocessing & Epoching Parameters (MUST MATCH TRAINING) ---
SAMPLING_RATE_HZ = 200.0
EPOCH_TMIN       = -0.2
EPOCH_TMAX       = 1.0

# --- Output Files ---
OUTPUT_PREDICTIONS_CSV = "test_predictions.csv"
OUTPUT_SUBMISSION_CSV = "submission.csv"
    
def process_single_file_to_features(file_key: str, group: pd.DataFrame, eeg_dir: str) -> Optional[pd.DataFrame]:
    eeg_csv_path = os.path.join(eeg_dir, f"{file_key}.csv")
    if not os.path.exists(eeg_csv_path):
        print(f"[WARN] Could not find EEG CSV for '{file_key}'. Skipping.")
        return None
    try:
        df_cols = pd.read_csv(eeg_csv_path, nrows=1).columns
        ALL_EEG_CHANNELS = [col for col in df_cols if col not in ['Time', 'FeedBackEvent', 'EOG']]
    except Exception as e:
        print(f"[ERROR] Could not read columns from {eeg_csv_path}: {e}")
        return None
    epochs = create_epochs_from_csv(eeg_csv_path, group, sfreq=SAMPLING_RATE_HZ, all_eeg_chans=ALL_EEG_CHANNELS, eog_chans=EOG_CHANNEL_COLS, feature_chans=EEG_CHANNEL_COLS, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
    if epochs is None or len(epochs) == 0:
        print(f"[WARN] No valid epochs were created for '{file_key}'. Skipping.")
        return None
    feats = extract_bci_features(epochs)
    processed_df = epochs.metadata.merge(feats, left_index=True, right_index=True)
    return processed_df

# ==============================================================================
# END: Reusable Processing & Feature Extraction Functions
# ==============================================================================


def run_prediction_pipeline():
    """
    Main function to load the model and predict on the test set.
    """
    # 1. LOAD THE TRAINED MODEL AND SCALER
    try:
        print("Loading pre-trained model and scaler...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Model or scaler not found. Please run the training script first.")
        return

    # 2. LOAD AND PREPARE TEST EVENT MAPPING
    try:
        test_feedback_df = pd.read_csv(TEST_FEEDBACK_MAPPING_CSV)
        test_feedback_df["_file_key"] = test_feedback_df[FILE_COL].str.replace(r'\.csv$', '', regex=True)
    except FileNotFoundError:
        print(f"[ERROR] Test mapping file not found at: {TEST_FEEDBACK_MAPPING_CSV}")
        return

    all_predictions = []
    feature_cols = [f'FRN_{ch}_mean_200_300ms' for ch in ['FCz', 'Cz']] + \
                   ['P300_Pz_mean_300_600ms', 'Theta_4_8Hz_power_FCz_200_400ms']

    # 3. LOOP THROUGH EACH TEST FILE AND MAKE PREDICTIONS
    print(f"\nFound {test_feedback_df['_file_key'].nunique()} test files to process.")
    for file_key, group in test_feedback_df.groupby("_file_key"):
        print(f"\n--- Processing test file: {file_key} ---")
        
        # a. Process the raw EEG file to get features using the reusable function
        feature_df = process_single_file_to_features(file_key, group, eeg_dir=TEST_EEG_DIR)
        
        if feature_df is None or feature_df.empty:
            print(f"Could not generate features for {file_key}. Skipping.")
            continue
            
        # b. Extract feature columns in the correct order
        X_new = feature_df[feature_cols]
        
        # c. Scale the new features using the LOADED scaler
        X_new_scaled = scaler.transform(X_new)
        
        # d. Make predictions
        predictions = model.predict_proba(X_new_scaled)[:, 1]
        
        # e. Store predictions with original info
        result_df = feature_df.copy() # feature_df contains metadata from kept epochs
        result_df['Prediction'] = predictions
        all_predictions.append(result_df)
        
        print(f"Successfully made {len(predictions)} predictions for this file.")

    # 4. COMBINE AND SAVE ALL PREDICTIONS
    if all_predictions:
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Save a detailed file for your own analysis
        final_predictions_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
        print(f"\nDetailed predictions saved to: {OUTPUT_PREDICTIONS_CSV}")

        # Save a simple submission file (e.g., for Kaggle)
        submission_df = final_predictions_df[['Feedback_ID', 'Prediction']]
        submission_df.to_csv(OUTPUT_SUBMISSION_CSV, index=False)
        print(f"Submission file saved to: {OUTPUT_SUBMISSION_CSV}")
        
        print("\n--- First 5 Predictions ---")
        print(submission_df.head())
    else:
        print("\nNo predictions were made. Check for warnings above.")


if __name__ == "__main__":
    run_prediction_pipeline()

