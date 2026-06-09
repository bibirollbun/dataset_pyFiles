import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from scipy import signal
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Ayarlar
TARGET_SR = 30
DATA_DIR_TRAIN = "../input/datasetfile/neymark-signal-processing/watch_data_train/watch_data_train"
DATA_DIR_TEST = "../input/datasetfile/neymark-signal-processing/watch_data_test/watch_data_test"

# --- Yardımcı Fonksiyonlar ---
def normalize_signal(signal_array):
    signal_array = (signal_array - np.min(signal_array)) / (np.max(signal_array) - np.min(signal_array))
    return signal_array - np.mean(signal_array)

def butter_bandpass_filter(data, lowcut=1.0, highcut=2.5, fs=30, order=3):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return signal.filtfilt(b, a, data)

def detect_peaks(signal_array, time_array):
    dynamic_thresh = np.mean(signal_array) + 0.3 * np.std(signal_array)
    peaks, _ = signal.find_peaks(signal_array, height=dynamic_thresh, distance=15)
    return time_array[peaks]

def compute_ibi_interpolated(peak_times, start, end, sr=30):
    duration = end - start
    resampled_time = np.linspace(start, end, int(duration * sr))
    intervals = np.diff(peak_times)
    ibi = np.interp(resampled_time[:-1], peak_times[1:], intervals)
    return resampled_time[:-1], ibi

def average_pooling(arr, k_size, pad=False):
    tensor = torch.tensor(arr, dtype=torch.float32).reshape(1, 1, -1)
    pooled = F.avg_pool1d(tensor, kernel_size=k_size, stride=1, padding=k_size // 2, count_include_pad=pad)
    return pooled.squeeze().numpy()[:-1]

def smooth_with_heuristic(signal_array):
    win = 300
    temp = 0
    high_th = 1.0
    low_th = 0.1
    for i in range(len(signal_array)):
        if temp < win:
            temp += 1
            continue
        diff = np.abs(np.mean(signal_array[i - win:i]) - signal_array[i])
        if low_th < diff < high_th:
            signal_array[i] = np.mean(signal_array[i - win // 10:i])
    return signal_array

def extract_time_series_features(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    features = {
        "mean_rr": np.mean(rr_intervals),
        "std_rr": np.std(rr_intervals),
        "rmssd": np.sqrt(np.mean(diff_rr**2)),
        "pnn50": np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr),
        "min_rr": np.min(rr_intervals),
        "max_rr": np.max(rr_intervals)
    }
    return features

def process_ppg_file(file_path, lowcut=1.0, highcut=2.5, visualize=False):
    df = pd.read_csv(file_path)
    time_vals = df['Timestamp'].values
    signal_vals = normalize_signal(df['PPG_signal'].values)
    filtered = butter_bandpass_filter(signal_vals, lowcut, highcut)
    peaks = detect_peaks(filtered, time_vals)
    interp_time, ibi = compute_ibi_interpolated(peaks, time_vals[0], time_vals[-1])
    ibi = average_pooling(ibi, k_size=40)
    ibi = smooth_with_heuristic(ibi)
    ibi = ibi[ibi < np.percentile(ibi, 95)]

    min_len = min(len(interp_time), len(ibi))
    interp_time = interp_time[:min_len]
    ibi = ibi[:min_len]

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_vals, signal_vals, label="Raw PPG", alpha=0.4)
        plt.plot(time_vals, filtered, label="Filtered", color='orange')
        plt.scatter(peaks, np.ones_like(peaks) * 0.5, color='red', marker='x', label="Peaks")
        plt.legend()
        plt.title("PPG Sinyali ve Tespit Edilen Zirveler")

        plt.subplot(2, 1, 2)
        plt.plot(interp_time, ibi, label="Interpolated IBI", color='purple')
        plt.title("Interbeat Interval (IBI)")
        plt.xlabel("Time (s)")
        plt.ylabel("IBI (s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ibi, interp_time

def process_folder(folder_path, output_csv, lowcut=1.0, highcut=2.5, visualize=False):
    ppg_files = sorted(glob.glob(os.path.join(folder_path, "PPG_EXP_*.csv")))
    all_ibi = []
    interp_times = []
    for i, file in enumerate(tqdm(ppg_files, desc=os.path.basename(folder_path))):
        ibi, interp_time = process_ppg_file(file, lowcut, highcut, visualize=(visualize and i == 0))
        all_ibi.extend(ibi)
        interp_times.extend(interp_time)

    df_out = pd.DataFrame({"PPG_interbeat_interval": all_ibi, "id": np.arange(len(all_ibi))})
    df_out.to_csv(output_csv, index=False)
    print(f"Kaydedildi: {output_csv}")
    return all_ibi, interp_times

# Train ve Test klasörlerini işle
ibi_train, interp_time_train = process_folder(DATA_DIR_TRAIN, "train_ppg_submission.csv", visualize=True)
ibi_test, interp_time_test = process_folder(DATA_DIR_TEST, "test_ppg_submission.csv", visualize=False)

# ECG karşılaştırması
def detect_ecg_peaks(ecg_df):
    times = ecg_df['ECG_peaks'].values
    peak_times = times[ecg_df.index]
    return compute_ibi_interpolated(peak_times, times[0], times[-1])[1]

ecgs = sorted(glob.glob(os.path.join(DATA_DIR_TRAIN, "ECG_EXP_*.csv")))
ecgs_ibi = []
for ecg_file in tqdm(ecgs, desc="ECG Files"):
    df_ecg = pd.read_csv(ecg_file)
    ibi = detect_ecg_peaks(df_ecg)
    ecgs_ibi.extend(ibi)

# Performans değerlendirme
min_len = min(len(ibi_train), len(ecgs_ibi))
rmse_score = mean_squared_error(ecgs_ibi[:min_len], ibi_train[:min_len], squared=False)
corr = np.corrcoef(ibi_train[:min_len], ecgs_ibi[:min_len])[0, 1]
print("RMSE:", rmse_score)
print("Correlation:", corr)

# === KAGGLE SUBMISSION DOSYASI ===
expected_rows = 1347666
current_len = len(ibi_test)

# Eksikse son değeri tekrar ederek doldur
if current_len < expected_rows:
    last_val = ibi_test[-1] if len(ibi_test) > 0 else 0.5  # yedek değer
    padding = [last_val] * (expected_rows - current_len)
    ibi_test.extend(padding)

# Fazlaysa kes
ibi_test = ibi_test[:expected_rows]

# Submission dosyasını oluştur
submission = pd.DataFrame({
    "id": np.arange(expected_rows),
    "PPG_interbeat_interval": ibi_test
})

submission.to_csv("/kaggle/working/submission.csv", index=False)

print("submission.csv ilk 5 satır:")
print(submission.head())
print("Toplam satır sayısı:", len(submission))


