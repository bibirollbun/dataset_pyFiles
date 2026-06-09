# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import os

train_data=  pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')


train_data.head()


import librosa
import sklearn
import json
import matplotlib.pyplot as plt
import wandb
import torchaudio
import plotly.express as px
from IPython.display import Audio
from shapely.geometry import Point
import plotly.express as px


base_dir = "/kaggle/input/birdclef-2025/train_audio"
data, sampling_rate = torchaudio.load(os.path.join(base_dir, train_data['filename'][3]))
data


display(Audio(data[0, :sampling_rate*100], rate=sampling_rate))


plt.figure(figsize=(20, 5))
plt.plot(data[0, :sampling_rate*6])
plt.show()


import scipy
from scipy import signal as sp_signal
def butter_highpass_filter(data, cutoff=300, fs=sampling_rate, order=10):
    normal_cutoff = cutoff / (fs / 2)
    b, a = sp_signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = sp_signal.filtfilt(b, a, data)
    return y


import torchaudio.functional as T

filtered_audio = T.highpass_biquad(waveform=data[:sampling_rate*40],sample_rate=32000, cutoff_freq=15950)  # remove low speechy hum
display(Audio(filtered_audio, rate=sampling_rate))


def fft_low_pass(data, cutoff, rate):
    stft = librosa.stft(data,  n_fft =int(rate * 0.093), hop_length = 128 )
    print(stft.shape)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=stft.shape[0] * 2 - 1)
    stft[freqs < cutoff, :] = 0  
    
    filtered_audio = librosa.istft(stft)
    
    return filtered_audio


data_np = data[0].numpy()
low_fft_filtered_data  = fft_low_pass(data_np[:sampling_rate*6], cutoff=2800,rate= sampling_rate)
plt.figure(figsize=(20, 5))
plt.plot(low_fft_filtered_data)
plt.show()


display(Audio(low_fft_filtered_data, rate=sampling_rate))


!pip install tensorflow librosa noisereduce matplotlib opencv-python pandas


# Cell: Check for CUDA/GPU Availability
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) found:")
    for gpu in gpus:
        print("  ", gpu)
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on GPU(s).")
    except RuntimeError as e:
        print("Error enabling memory growth:", e)
else:
    print("No GPU found. Using CPU.")


import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import noisereduce as nr
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_audio_file(file_path, sr=None):
    """Load an audio file and return the audio time series and sample rate."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def compute_spectrogram(audio, sr, n_fft=2048, hop_length=512):
    """Compute a spectrogram (in dB) from an audio signal."""
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def save_spectrogram(S_db, sr, filename_prefix):
    """Save the spectrogram as an image and as a NumPy array."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{filename_prefix} Spectrogram')
    plt.savefig(f'{filename_prefix}_spectrogram.png')
    plt.close()
    
    np.save(f'{filename_prefix}_spectrogram.npy', S_db)

def build_noise_profile(noise_dir, sr=None):
    """Load all .ogg noise recordings from a directory and concatenate them into a combined noise profile."""
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.ogg')]
    noise_profiles = []
    
    for nf in noise_files:
        audio, file_sr = load_audio_file(nf, sr=sr)
        noise_profiles.append(audio)
    
    combined_noise = np.concatenate(noise_profiles)
    return combined_noise

def resize_spectrogram(S_db, target_shape=(256, 256)):
    """Resize the spectrogram to a fixed target shape using OpenCV."""
    S_db_resized = cv2.resize(S_db.astype(np.float32), target_shape, interpolation=cv2.INTER_AREA)
    return S_db_resized

def create_training_pair(file_path, noise_profile, sr=None, target_shape=(256,256)):
    """Generate a (noisy spectrogram, denoised spectrogram) pair for a given audio file."""
    # Load the animal sound audio
    audio, sr = load_audio_file(file_path, sr=sr)
    
    # Compute and resize the original (noisy) spectrogram
    S_db_noisy = compute_spectrogram(audio, sr)
    S_db_noisy = resize_spectrogram(S_db_noisy, target_shape)
    
    # Apply noise reduction using the combined noise profile
    reduced_audio = nr.reduce_noise(audio_clip=audio, noise_clip=noise_profile, verbose=False)
    S_db_denoised = compute_spectrogram(reduced_audio, sr)
    S_db_denoised = resize_spectrogram(S_db_denoised, target_shape)
    
    # Expand dimensions to add a channel (grayscale image)
    S_db_noisy = np.expand_dims(S_db_noisy, axis=-1)
    S_db_denoised = np.expand_dims(S_db_denoised, axis=-1)
    
    return S_db_noisy, S_db_denoised


# Set paths (adjust these to your environment)
train_audio_dir = 'birdclef-2025/train_audio' 
train_soundscapes_dir = 'birdclef-2025/train_soundscapes'
csv_path = 'birdclef-2025/train.csv'  # Optional; use for later classification steps


import glob

# Collect all .ogg files from any subfolder within train_audio_dir
train_audio_files = glob.glob(os.path.join(train_audio_dir, '**/*.ogg'), recursive=True)
print(f"Found {len(train_audio_files)} .ogg files in train_audio (including subfolders).")

# Pick a sample animal sound from train_audio_files
if train_audio_files:
    sample_audio_path = train_audio_files[0]
    audio, sr = load_audio_file(sample_audio_path, sr=44100)
    print(f"Sample audio loaded: {len(audio)/sr:.2f} seconds at {sr} Hz")
    S_db_before = compute_spectrogram(audio, sr)
    save_spectrogram(S_db_before, sr, 'sample_before_noise_reduction')
else:
    print("No sample audio found in train_audio.")

