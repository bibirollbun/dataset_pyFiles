# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass
        
print("Load process is finished")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
import shutil
import soundfile as sf
import torchaudio
import random
import IPython.display as ipd
import tqdm


train_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
train_df


file_path = AUDIO_BASE_DIR / "1462737/CSA36369.ogg"
print(f"\nPlaying directly: {file_path}")
display(ipd.Audio(str(file_path), rate=32000))
print('Audio with load')
wav, sr = librosa.load(file_path, sr = 32000)
display(ipd.Audio(wav, rate=sr))


import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import IPython.display as ipd
import pandas as pd
import soundfile as sf

# === Load VAD ===
torch.set_num_threads(1)
model, (get_speech_timestamps, _, _, _, _) = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', 
    model='silero_vad',
    trust_repo=True
)

# === Settings ===
AUDIO_BASE_DIR = Path("/kaggle/input/birdclef-2025/train_audio")
AUTHOR_FILTER = "Fabio A. Sarria-S"
CHUNK_LEN = 0.1  # seconds
NUM_AUDIO_TO_DISPLAY = 26  # display limit

# === Load your dataframe ===
# train_df = pd.read_csv("path/to/train.csv")  # Uncomment and adjust if needed
fabio_df = train_df[train_df['author'] == AUTHOR_FILTER].copy()
fabio_df_len = len(fabio_df)
fabio_df.reset_index(drop=True, inplace=True)


NUM_AUDIO_TO_DISPLAY = min(NUM_AUDIO_TO_DISPLAY, fabio_df_len)
print(f"Total files to be displayed: {NUM_AUDIO_TO_DISPLAY}")
# === Iterate and visualize ===
for i, row in fabio_df.iterrows():
    file_path = AUDIO_BASE_DIR / row['filename']
    if not file_path.exists():
        print(f"Missing file: {file_path}")
        continue

    # Load audio
    wav_full, sr = librosa.load(file_path, sr=16000)
    wav = wav_full.copy()

    # Detect speech
    speech_timestamps = get_speech_timestamps(torch.Tensor(wav), model)

    # Truncate audio from start of first detected speech
    if speech_timestamps:
        first_start = speech_timestamps[0]['start']
        wav = wav[:first_start]

    # Compute audio power in dB (for detection only)
    chunk = int(CHUNK_LEN * sr)
    power = wav ** 2
    pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
    power = np.pad(power, (0, pad))
    power_chunks = power.reshape((-1, chunk)).sum(axis=1)
    power_db = 10 * np.log10(power_chunks + 1e-6)

    # Use mean of valid chunks instead of max to define threshold
    mean_db = np.mean(power_db)
    threshold = mean_db - 10  # 10 dB below mean plateau
    valid_mask = power_db >= threshold

    if np.any(valid_mask):
        start_chunk = np.argmax(valid_mask)
        end_chunk = len(valid_mask) - np.argmax(valid_mask[::-1])
        power_db_trimmed = power_db[start_chunk:end_chunk]

        # Detect trailing silence using rolling window on power_db
        window_size = 3
        silence_threshold = threshold
        binary_silence = (power_db_trimmed < silence_threshold).astype(int)
        rolling = np.convolve(binary_silence, np.ones(window_size, dtype=int), mode='valid')

        silence_start_rel = np.argmax(rolling == window_size)
        if rolling[silence_start_rel] == window_size:
            end_chunk = start_chunk + silence_start_rel

        start_sample = start_chunk * chunk
        end_sample = end_chunk * chunk
        
        start_time_sec = round(start_sample / sr, 1)
        end_time_sec = round(end_sample / sr, 1)
        
        start_chunk = int(start_sample / chunk)
        start_chunk = int(end_sample / chunk)
        
        wav = wav_full[start_sample:end_sample]  # Slice original waveform cleanly
        power_db = power_db[start_chunk:end_chunk]

        print(f"ROI start: {start_time_sec:.2f}s, end: {end_time_sec:.2f}s")


    # Create dummy segmentation (voice already removed)
    segmentation = np.zeros_like(wav)

    # limit number of displayed
    print(f"Showing file {i+1}")
    if i < NUM_AUDIO_TO_DISPLAY:
        # Plot
        fig = plt.figure(figsize=(24, 3))
        fig.suptitle(f"{row['filename']} by {row['author']}")
        t_power = np.arange(len(power_db)) * CHUNK_LEN
        plt.plot(t_power, power_db, 'b', label='Audio Power (dB)')
        t_seg = np.arange(len(segmentation)) / sr
        plt.plot(t_seg, segmentation, 'r', label='Voice Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (dB) / Voice Detection')
        plt.legend()
        plt.show()

        # Play audio
        display(ipd.Audio(wav, rate=sr))


