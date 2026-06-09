import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from torch.utils.data import Dataset


class Config:
    BASE_PATH = '/kaggle/input/waveform-inversion/'
    N_SAMPLES_PER_FILE = 500
    SENSOR_GRID = (7, 10) 

class PATHS:
    train = f'{Config.BASE_PATH}/train_samples'



class FileManager:
    @staticmethod
    def input_to_output(files):
        return [
            Path(str(f).replace('seis','vel').replace('data','model')) for f in files
        ]

    @staticmethod
    def get_file_lists():
        all_inputs = [f for f in Path(PATHS.train).rglob('*.npy') if ('seis' in f.stem) or ('data' in f.stem)]
        train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)]
        valid_inputs = [f for f in all_inputs if f not in train_inputs]

        train_outputs = FileManager.input_to_output(train_inputs)
        valid_outputs = FileManager.input_to_output(valid_inputs)

        return train_inputs, train_outputs, valid_inputs, valid_outputs



class SeismicDataset(Dataset):
    def __init__(self, input_files, output_files, n_samples_per_file=500):
        self.input_files = input_files
        self.output_files = output_files
        self.n_samples_per_file = n_samples_per_file

    def __len__(self):
        return len(self.input_files) * self.n_samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.n_samples_per_file
        sample_idx = idx % self.n_samples_per_file
        
        X = np.load(self.input_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y


train_inputs, train_outputs, valid_inputs, valid_outputs = FileManager.get_file_lists()
dataset = SeismicDataset(train_inputs, train_outputs)
x, y = dataset[1000]


plt.figure(figsize=(6, 6))
plt.imshow(y[0], cmap='seismic', origin='lower')
plt.colorbar(label='Amplitude')
plt.title('Seismic Field (1 Channel, 70x70 Grid)')
plt.xlabel('Receiver Index')
plt.ylabel('Source Index')
plt.tight_layout()
plt.show()


heatmap = np.squeeze(y)

for ch in range(5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for sensor in range(70):
        axs[0].plot(x[ch, :, sensor], alpha=0.4)
    axs[0].set_title(f'All Sensors - Channel {ch}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    im = axs[1].imshow(heatmap, cmap='viridis', origin='lower')
    axs[1].set_title('Sensor Grid Heatmap (Target)')
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Channel {ch} - Time Series & Heatmap')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



time_index = 800
for channel in range(5):
    snapshot = x[channel, time_index, :].reshape(Config.SENSOR_GRID)
    plt.figure(figsize=(6, 5))
    plt.imshow(snapshot, cmap='seismic', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title(f'2D Sensor Grid at Time {time_index} - Channel {channel}')
    plt.xlabel('Sensor X')
    plt.ylabel('Sensor Y')
    plt.tight_layout()
    plt.show()



from pprint import pprint
class SensorFeatureExtractor:
    @staticmethod
    def compute_features(x, fs=100):
        C, T, S = x.shape
        stats = {}

        for sensor in range(S):
            stats[f"Sensor_{sensor}"] = {}
            for ch in range(C):
                sig = x[ch, :, sensor]

                # Time domain
                mean_val = np.mean(sig)
                std_val = np.std(sig)
                rms = np.sqrt(np.mean(sig ** 2))
                energy = np.sum(sig ** 2)
                peak = np.max(np.abs(sig))
                zcr = ((sig[:-1] * sig[1:]) < 0).sum()
                skewness = skew(sig)
                kurt_val = kurtosis(sig)

                # Frequency domain
                freqs, psd = welch(sig, fs=fs)
                psd_norm = psd / np.sum(psd)
                spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))

                def band_power(fmin, fmax):
                    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                    return np.sum(psd[idx])

                bp_0_5_3 = band_power(0.5, 3)
                bp_3_8 = band_power(3, 8)
                bp_8_20 = band_power(8, 20)

                stats[f"Sensor_{sensor}"][f"Channel_{ch}"] = {
                    "mean": mean_val,
                    "std": std_val,
                    "rms": rms,
                    "energy": energy,
                    "peak_amplitude": peak,
                    "zero_crossings": zcr,
                    "skewness": skewness,
                    "kurtosis": kurt_val,
                    "spectral_entropy": spec_entropy,
                    "bandpower_0.5-3Hz": bp_0_5_3,
                    "bandpower_3-8Hz": bp_3_8,
                    "bandpower_8-20Hz": bp_8_20
                }

        return stats

features = SensorFeatureExtractor.compute_features(x)
pprint(features)


