!pip install librosa


import os
import glob

import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


ogg_files = glob.glob("/kaggle/input/birdclef-2025/train_audio/**/*.ogg", recursive=True)
print(f"音声ファイル数：{len(ogg_files)}")


min_duration = 5.0 # 分析対象とする最小の長さ（秒）

five_sec_plus_ogg_files = []
for path in tqdm(ogg_files):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration >= min_duration:
            five_sec_plus_ogg_files.append(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
five_sec_plus_ogg_file_ratio = len(five_sec_plus_ogg_files) / len(ogg_files) * 100

print(f"{min_duration}秒以上の音声ファイル数: {len(five_sec_plus_ogg_files)} / {len(ogg_files)} ({five_sec_plus_ogg_file_ratio:.2f}%)")


rms_ratios = []

for path in tqdm(five_sec_plus_ogg_files):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # 中央5秒を切り出し
        start = int((duration / 2 - 2.5) * sr)
        end = int((duration / 2 + 2.5) * sr)
        y_center = y[start:end]

        # RMS計算
        rms_total = librosa.feature.rms(y=y)[0]
        avg_rms_total = np.mean(rms_total)

        rms_center = librosa.feature.rms(y=y_center)[0]
        avg_rms_center = np.mean(rms_center)

        # 比率を記録
        if avg_rms_total > 0:
            ratio = avg_rms_center / avg_rms_total
            rms_ratios.append(ratio)

    except Exception as e:
        print(f"Error with {path}: {e}")


plt.hist(rms_ratios, bins=50)
plt.axvline(x=1.0, color='red', linestyle='--', label="Center = Total")
plt.xlabel("Center RMS / Total RMS")
plt.ylabel("Number of files")
plt.title("RMS Ratio: Center 5s vs Entire Clip")
plt.grid(True)
plt.legend()
plt.show()


ratios = np.array(rms_ratios)
print(f"サンプル数: {len(ratios)}")
print(f"平均: {np.mean(ratios):.3f}")
print(f"中央値: {np.median(ratios):.3f}")
print(f"0.8未満の割合: {(np.sum(ratios < 0.8) / len(ratios)) * 100:.1f}%")
print(f"0.5未満の割合: {(np.sum(ratios < 0.5) / len(ratios)) * 100:.1f}%")

