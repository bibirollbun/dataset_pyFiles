import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ground_truth = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
ground_truth = ground_truth.iloc[11].values
planet_id = int(ground_truth[0])
ground_truth = ground_truth[2:]
ground_truth = ground_truth - np.mean(ground_truth)

airs_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_0.parquet', engine='pyarrow').values.reshape((11250 // 2, 2, 32, 356))

airs_signal = airs_signal / 0.4369 - 1000
airs_signal = airs_signal[:, 1, ...] - airs_signal[:, 0, ...]

low = np.nanmean(airs_signal[2000: 2500, :, 39: 321], axis=(0, 1))
high = np.nanmean(airs_signal[4000: 5000, :, 39: 321], axis=(0, 1))

spectrum = 1 - low / high
spectrum = pd.Series(spectrum).rolling(21, min_periods=1, center=True).mean()
spectrum = spectrum - np.mean(spectrum)

plt.figure('Bug?')

plt.plot(np.arange(282), spectrum, label='Naive AIRS spectrum')
plt.plot(np.arange(282), ground_truth, label='Original ground truth')
plt.plot(np.arange(282), np.flip(ground_truth), label='Flipped ground truth')

plt.legend()

plt.show()


