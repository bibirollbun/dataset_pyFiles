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
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install numpy librosa soundfile torch torchaudio tensorflow tensorflow-io audiomentations


import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


DATA_ROOT = "birdclef-2025" 
TRAIN_METADATA_PATH = os.path.join(DATA_ROOT, "train_metadata.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_ROOT, "train_audio")

# Load the training metadata
try:
    train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
    print("Training metadata loaded successfully.")
    print(f"Number of training examples: {len(train_df)}")
except FileNotFoundError:
    print(f"Error: Training metadata file not found at {TRAIN_METADATA_PATH}. "
          "Please ensure the dataset is downloaded and the path is correct.")
    train_df = None


if train_df is not None:
    print("\n--- Basic Information about the Training Data ---")
    print(train_df.head())
    print(train_df.info())
    print(train_df.describe())


    print("\n--- Exploring the Target Variable (Species) ---")
    print(f"Number of unique bird species: {train_df['primary_label'].nunique()}")
    species_counts = train_df['primary_label'].value_counts().sort_values(ascending=False)
    print("\nTop 10 most frequent species:\n", species_counts.head(10))

    # Plot the distribution of the top N species
    top_n = 20
    plt.figure(figsize=(12, 6))
    species_counts.head(top_n).plot(kind='bar')
    plt.title(f"Distribution of Top {top_n} Bird Species")
    plt.xlabel("Bird Species")
    plt.ylabel("Number of Recordings")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    print("\n--- Exploring Audio Files ---")

    # Example: Load and listen to a random audio file
    random_audio_path = os.path.join("/kaggle/input/birdclef-2025/train.csv",
                                     train_df.sample(1)['primary_label'].iloc[0],
                                     train_df.sample(1)['filename'].iloc[0])
    print(f"Loading and playing: {random_audio_path}")
    try:
        audio, sr = librosa.load("/kaggle/input/birdclef-2025/train_audio/greani1/XC132190.ogg", sr=None)  # Load with original sampling rate
        print(f"Shape of audio: {audio.shape}")
        print(f"Sampling rate: {sr} Hz")
        ipd.display(ipd.Audio(audio, rate=sr))

        # Visualize the waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

        # Visualize the spectrogram
        n_fft = 2048
        hop_length = 512
        stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft_result)
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram (Log Scale)")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Audio file not found at {random_audio_path}")
    except Exception as e:
        print(f"Error loading audio file: {e}")


    print("\n--- Exploring Audio File Durations ---")
    durations = []
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Analyzing audio durations"):
        audio_path = os.path.join("/kaggle/input/birdclef-2025/train_audio/greani1", row['primary_label'], row['filename'])
        try:
            audio, sr = librosa.load("/kaggle/input/birdclef-2025/train_audio/greani1/XC132190.ogg", sr=None, duration=None)  # Load full duration
            durations.append(librosa.get_duration(y=audio, sr=sr))
        except Exception as e:
            durations.append(np.nan)
            print(f"Error loading {audio_path}: {e}")

    train_df['duration'] = durations
    print(train_df['duration'].describe())

    # Plot the distribution of audio durations
    plt.figure(figsize=(8, 6))
    plt.hist(train_df['duration'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Audio Durations")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


    print("\n--- Further Exploration Ideas ---")
    print("- Explore the 'secondary_labels' column (if present in your dataset version).")
    print("- Investigate the geographical information (latitude, longitude) if available.")
    print("- Look for any correlations between metadata features.")
    print("- Explore the test data structure (if you are participating in a competition).")
    print("Skipping data exploration due to missing training metadata.")


def load_audio(audio_path, target_sr=None, duration=None):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
        if duration is not None:
            target_samples = int(duration * sr)
            if len(y) < target_samples:
                padding = target_samples - len(y)
                y = np.pad(y, (0, padding), 'constant')
            elif len(y) > target_samples:
                y = y[:target_samples]
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None


def extract_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    if audio is None:
        return None
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def process_audio_file(audio_path, target_sr=32000, duration=5.0,
                       n_fft=1024, hop_length=512, n_mels=64):
    audio, sr = load_audio(audio_path, target_sr=target_sr, duration=duration)
    if audio is not None:
        features = extract_mel_spectrogram(audio, sr, n_fft, hop_length, n_mels)
        return features
    return None


def process_directory(audio_dir, output_dir, target_sr=32000, duration=5.0,
                      n_fft=1024, hop_length=512, n_mels=64):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.ogg', '.flac', '.mp3')):  # Add more extensions if needed
            audio_path = os.path.join(audio_dir, filename)
            features = process_audio_file(audio_path, target_sr, duration,
                                          n_fft, hop_length, n_mels)
            if features is not None:
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}.npy")
                np.save(output_path, features)
                print(f"Processed and saved features for {filename} to {output_path}")


if __name__ == '__main__':
    # Example usage:

    # 1. Process a single audio file
    audio_file_path = '/kaggle/input/birdclef-2025/train_audio/greani1/XC132190.ogg'  # Replace with the actual path
    mel_spectrogram = process_audio_file(audio_file_path)
    if mel_spectrogram is not None:
        print("Mel spectrogram shape for single file:", mel_spectrogram.shape)
        # You can now use this 'mel_spectrogram' for further tasks

    # 2. Process all audio files in a directory
    audio_directory = '/kaggle/input/birdclef-2025/train_audio/'  # Replace with the actual path to your audio directory
    output_directory = '/kaggle/input/birdclef-2025/train_audio/'  # Replace with the desired output directory
    process_directory(audio_directory, output_directory)
    print(f"Processed all audio files in {audio_directory} and saved features to {output_directory}")


def visualize_mel_spectrogram(mel_spectrogram_db, sr, title="Mel Spectrogram"):
    if mel_spectrogram_db is None:
        print("No Mel spectrogram to visualize.")
        return

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_features(mfccs, chroma, sr, title="Audio Features"):
    if mfccs is None or chroma is None:
        print("No features to visualize.")
        return

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCCs')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Chroma Features')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.show()


def extract_features(audio_path, target_sr=None, duration=None, n_fft=2048, hop_length=512, n_mfcc=20, n_chroma=12):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)

        return mfccs, chroma, sr

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None, None, None


if __name__ == '__main__':
    audio_file = '/kaggle/input/birdclef-2025/train_audio/greani1/XC556246.ogg'  # Replace with the actual path to your audio file

    mfccs, chroma, sr = extract_features(audio_file, target_sr=32000, duration=5.0, n_mfcc=20, n_chroma=12)

    if mfccs is not None and chroma is not None:
        print("Shape of MFCCs:", mfccs.shape)
        print("Shape of Chroma features:", chroma.shape)
        print("Sampling rate:", sr)

        visualize_features(mfccs, chroma, sr, title=f"Features for {audio_file}")


pip install --upgrade audiomentations


import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import tensorflow as tf
import tensorflow_io as tfio
#from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift, Gain, SpecAugment
#from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift, Gain, FrequencyMask, TimeMask
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift, Gain # waveform transforms
import random
import matplotlib.pyplot as plt
import os


# --- Configuration (Adjust as needed) ---
SAMPLE_RATE = 32000  # BirdCLEF datasets often use 32kHz or higher
DURATION = 5       # Standard clip duration (e.g., 5 seconds for BirdCLEF segments)
N_MELS = 128       # Number of mel bins for spectrograms
N_FFT = 2048       # FFT window size
HOP_LENGTH = 512   # Hop length for STFT


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Loads an audio file and resamples/pads/trims it to a target duration."""
    try:
        y, current_sr = librosa.load(file_path, sr=None)
        if current_sr != sr:
            y = librosa.resample(y, orig_sr=current_sr, target_sr=sr)

        # Pad or trim to desired duration
        target_samples = int(duration * sr)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            y = y[:target_samples]
        return y, sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None, None


# --- Augmentation Functions (using librosa/numpy for waveform processing) ---

def time_stretch_audio(audio, sr, rate_range=(0.8, 1.2)):
    rate = random.uniform(*rate_range)
    # librosa.effects.time_stretch resamples, so target_sr is the original sr
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift_audio(audio, sr, n_steps_range=(-2, 2)):
    n_steps = random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise_audio(audio, noise_audio=None, snr_db_range=(3, 30)):
    target_snr_db = random.uniform(*snr_db_range)

    if noise_audio is None:
        # Generate Gaussian noise
        noise = np.random.randn(len(audio))
    else:
        # Use provided noise, pad/trim to match audio length
        if len(noise_audio) < len(audio):
            noise = np.pad(noise_audio, (0, len(audio) - len(noise_audio)), 'wrap') # Use wrap for continuous noise
        else:
            noise = noise_audio[:len(audio)]

    # Calculate signal and noise power
    P_signal = np.mean(audio**2)
    P_noise = np.mean(noise**2)

    # Calculate desired noise power for target SNR
    target_P_noise = P_signal / (10**(target_snr_db / 10))

    # Scale noise to desired power
    scaled_noise = noise * np.sqrt(target_P_noise / P_noise)

    return audio + scaled_noise

def time_shift_audio(audio, sr, shift_range=(-0.5, 0.5)):
    shift_seconds = random.uniform(*shift_range)
    shift_samples = int(shift_seconds * sr)
    
    # Use np.roll for circular shift, effectively moving parts from one end to the other
    return np.roll(audio, shift_samples)

def volume_adjustment_audio(audio, gain_db_range=(-6, 6)):
    gain_db = random.uniform(*gain_db_range)
    gain_amplitude = 10**(gain_db / 20) # Convert dB to amplitude multiplier
    return audio * gain_amplitude


# --- Spectrogram-based Augmentations (SpecAugment) ---

def apply_spec_augment(mel_spectrogram, time_mask_param=40, freq_mask_param=20, num_time_masks=1, num_freq_masks=1):
    if isinstance(mel_spectrogram, np.ndarray):
        # Convert to torch tensor for torchaudio's SpecAugment
        mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0) # Add batch dimension
    else:
        mel_spectrogram_tensor = mel_spectrogram

    augmented_spec = mel_spectrogram_tensor.clone() # Work on a copy

    # Time Masking
    for _ in range(num_time_masks):
        mask_length = random.randint(0, time_mask_param)
        mask_start = random.randint(0, augmented_spec.shape[-1] - mask_length)
        augmented_spec[:, :, mask_start:mask_start + mask_length] = 0.0

    # Frequency Masking
    for _ in range(num_freq_masks):
        mask_length = random.randint(0, freq_mask_param)
        mask_start = random.randint(0, augmented_spec.shape[-2] - mask_length)
        augmented_spec[:, mask_start:mask_start + mask_length, :] = 0.0

    if isinstance(mel_spectrogram, np.ndarray):
        return augmented_spec.squeeze(0).numpy() # Remove batch dimension and convert back
    else:
        return augmented_spec


# --- Mixup ---

def mixup_spectrograms(spec1, label1, spec2, label2, alpha=0.2):
    lam = np.random.beta(alpha, alpha) # Lambda from Beta distribution
    mixed_spec = lam * spec1 + (1 - lam) * spec2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_spec, mixed_label


# --- Example Usage with a Dummy BirdCLEF-like Dataset Structure ---

class BirdCLEFAudioDataset:
    def __init__(self, audio_files, labels, sr=SAMPLE_RATE, duration=DURATION):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.noise_pool = self._load_noise_samples(sr=sr) # Load some background noise

    def _load_noise_samples(self, noise_dir="noise_samples", sr=SAMPLE_RATE):
        """Loads a few noise samples from a directory for background noise augmentation."""
        noise_samples = []
        if os.path.exists(noise_dir):
            for fname in os.listdir(noise_dir):
                if fname.endswith(('.wav', '.ogg', '.flac')):
                    noise_path = os.path.join(noise_dir, fname)
                    try:
                        noise, _ = librosa.load(noise_path, sr=sr)
                        noise_samples.append(noise)
                    except Exception as e:
                        print(f"Warning: Could not load noise file {noise_path}: {e}")
        print(f"Loaded {len(noise_samples)} noise samples from {noise_dir}")
        return noise_samples

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio (and handle resampling/padding/trimming)
        audio, sr = load_audio(file_path, self.sr, self.duration)
        if audio is None:
            # Handle loading error, e.g., return a placeholder or skip
            return np.zeros(int(self.duration * self.sr)), np.zeros_like(label) # Placeholder

        # Apply augmentations with a probability
        if random.random() < 0.7: # 70% chance to apply augmentations
            # Randomly select a subset of augmentations or apply them sequentially
            
            # 1. Time Stretching
            if random.random() < 0.5:
                audio = time_stretch_audio(audio, sr, rate_range=(0.9, 1.1)) # Slightly less aggressive stretching

            # 2. Pitch Shifting
            if random.random() < 0.5:
                audio = pitch_shift_audio(audio, sr, n_steps_range=(-1, 1)) # Slightly less aggressive shifting

            # 3. Adding Noise
            if random.random() < 0.6:
                noise_audio = random.choice(self.noise_pool) if self.noise_pool else None
                audio = add_noise_audio(audio, noise_audio=noise_audio, snr_db_range=(10, 25)) # Higher SNR means less noise

            # 4. Time Shifting
            if random.random() < 0.5:
                audio = time_shift_audio(audio, sr, shift_range=(-0.1, 0.1)) # Small shifts

            # 5. Volume Adjustment
            if random.random() < 0.6:
                audio = volume_adjustment_audio(audio, gain_db_range=(-4, 4))

        # Convert to Mel Spectrogram
        # For PyTorch
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) # Convert to dB scale

        # Apply SpecAugment on the spectrogram
        if random.random() < 0.7:
            mel_spectrogram = apply_spec_augment(
                mel_spectrogram,
                time_mask_param=int(mel_spectrogram.shape[1] * 0.1), # Mask up to 10% of time steps
                freq_mask_param=int(mel_spectrogram.shape[0] * 0.1), # Mask up to 10% of freq bins
                num_time_masks=2,
                num_freq_masks=2
            )
        
        # Mixup is usually applied at the batch level in the DataLoader
        # For simplicity, we'll return the spectrogram and label for now.
        return mel_spectrogram.astype(np.float32), label.astype(np.float32)


# --- Example using audiomentations (Recommended for waveform augmentations) ---

class BirdCLEFAudioDatasetAudiomentations(BirdCLEFAudioDataset):
    def __init__(self, audio_files, labels, sr=SAMPLE_RATE, duration=DURATION):
        super().__init__(audio_files, labels, sr, duration)
        self.augment = Compose([
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5, leave_length_unchanged=True),
            PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.5),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6), # Simulates moderate noise
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
            Gain(min_gain_db=-4.0, max_gain_db=4.0, p=0.6),
            # SpecAugment is applied after Mel Spectrogram conversion, not directly on waveform
        ])
        
    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[idx]

        audio, sr = load_audio(file_path, self.sr, self.duration)
        if audio is None:
            return np.zeros(int(self.duration * self.sr)), np.zeros_like(label)

        # Apply waveform augmentations
        augmented_audio = self.augment(samples=audio, sample_rate=sr)

        # Convert to Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=augmented_audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) # Convert to dB scale

        # Apply SpecAugment on the spectrogram
        if random.random() < 0.7:
            mel_spectrogram = apply_spec_augment(
                mel_spectrogram,
                time_mask_param=int(mel_spectrogram.shape[1] * 0.1),
                freq_mask_param=int(mel_spectrogram.shape[0] * 0.1),
                num_time_masks=2,
                num_freq_masks=2
            )
        
        return mel_spectrogram.astype(np.float32), label.astype(np.float32)


# --- Dummy Data Setup (replace with your actual BirdCLEF data loading) ---

# Create dummy audio files and labels for demonstration
dummy_audio_dir = "dummy_birdclef_audio"
noise_samples_dir = "noise_samples"
os.makedirs(dummy_audio_dir, exist_ok=True)
os.makedirs(noise_samples_dir, exist_ok=True)

dummy_labels = []
dummy_audio_paths = []
num_classes = 10
num_samples_per_class = 5
total_samples = num_classes * num_samples_per_class

print("Generating dummy audio and noise files...")
for i in range(total_samples):
    dummy_file = os.path.join(dummy_audio_dir, f"audio_{i:03d}.wav")
    y_dummy = np.random.randn(int(SAMPLE_RATE * DURATION)).astype(np.float32) * 0.5 # Random noise for dummy audio
    sf.write(dummy_file, y_dummy, SAMPLE_RATE)
    dummy_audio_paths.append(dummy_file)
    
    one_hot_label = np.zeros(num_classes)
    one_hot_label[i % num_classes] = 1 # Simple dummy one-hot label
    dummy_labels.append(one_hot_label)

# Create dummy noise files
for i in range(3):
    noise_file = os.path.join(noise_samples_dir, f"noise_{i}.wav")
    y_noise = np.random.randn(int(SAMPLE_RATE * DURATION * 2)).astype(np.float32) * 0.1 # Longer noise
    sf.write(noise_file, y_noise, SAMPLE_RATE)

print("Dummy data generation complete.")


# --- How to integrate into your training loop (PyTorch example) ---

# 1. Create your dataset instance
# dataset = BirdCLEFAudioDataset(dummy_audio_paths, dummy_labels)
dataset = BirdCLEFAudioDatasetAudiomentations(dummy_audio_paths, dummy_labels) # Using audiomentations

# 2. Create a DataLoader (for batching and shuffling)
from torch.utils.data import DataLoader

# A custom collate_fn is often needed for audio, especially if lengths vary slightly
# or if you plan to do mixup at the batch level.
def collate_fn_spectrogram(batch):
    # batch is a list of (spectrogram, label) tuples
    spectrograms = torch.stack([torch.from_numpy(item[0]) for item in batch])
    labels = torch.stack([torch.from_numpy(item[1]) for item in batch])
    return spectrograms, labels

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_spectrogram)

print("\n--- Demonstrating data augmentation with PyTorch DataLoader ---")
for batch_idx, (spectrograms, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"  Spectrograms shape: {spectrograms.shape}") # (batch_size, n_mels, n_frames)
    print(f"  Labels shape: {labels.shape}")             # (batch_size, num_classes)

    # Optional: Visualize an augmented spectrogram
    if batch_idx == 0:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrograms[0].numpy(), sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Augmented Mel Spectrogram Example')
        plt.tight_layout()
        plt.show()

    if batch_idx == 1: # Just show a couple of batches
        break


# --- TensorFlow Example (Conceptual) ---

# For TensorFlow, you would typically use tf.data.Dataset and `tf.py_function`
# or `tfio.audio` for augmentations.
# tensorflow_io provides some built-in augmentations like SpecAugment (frequency and time mask).

def preprocess_tf_audio(file_path, label):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = audio_tensor.to_tensor()[:, 0] # Get mono channel if stereo
    rate = audio_tensor.rate.numpy()

    # Resample/Pad/Trim (similar logic as load_audio, but with TF operations)
    audio = tf.cast(audio, tf.float32)
    current_length = tf.shape(audio)[0]
    target_length = tf.cast(SAMPLE_RATE * DURATION, tf.int32)
    if current_length < target_length:
        padding = target_length - current_length
        audio = tf.pad(audio, [[0, padding]], "CONSTANT")
    elif current_length > target_length:
        audio = audio[:target_length]

    # Example TensorFlow augmentations
    # Note: Waveform augmentations like time stretching/pitch shifting are more complex
    # directly in TensorFlow's graph mode without custom ops or tf.py_function.
    # tfio.audio provides some, but less comprehensive than librosa/audiomentations.
    
    # Volume Adjustment (Gain) in TensorFlow
    gain_db = tf.random.uniform([], minval=-4.0, maxval=4.0)
    gain_amplitude = tf.pow(10.0, gain_db / 20.0)
    audio = audio * gain_amplitude

    # Add Gaussian Noise (simple version)
    if tf.random.uniform([]) < 0.6:
        noise = tf.random.normal(tf.shape(audio), stddev=0.01) # Small stddev for subtle noise
        audio = audio + noise

    # Mel Spectrogram
    stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH)
    spectrogram = tf.abs(stft)
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=rate, lower_edge_hertz=20.0, upper_edge_hertz=rate/2)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6) # Log-scale

    # SpecAugment in TensorFlow
    if tf.random.uniform([]) < 0.7:
        mel_spectrogram = tfio.audio.freq_mask(mel_spectrogram, param=int(N_MELS * 0.1))
        mel_spectrogram = tfio.audio.time_mask(mel_spectrogram, param=int(mel_spectrogram.shape[1] * 0.1))

    return mel_spectrogram, label

# # Example of how to use with tf.data.Dataset (uncomment to run TF part)
# print("\n--- Demonstrating data augmentation with TensorFlow tf.data ---")
# # Assuming dummy_audio_paths and dummy_labels are populated
# tf_dataset = tf.data.Dataset.from_tensor_slices((dummy_audio_paths, dummy_labels))
# tf_dataset = tf_dataset.map(lambda x, y: tf.py_function(
#     preprocess_tf_audio, [x, y], (tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
# tf_dataset = tf_dataset.batch(4)
#
# for batch_idx, (spectrograms, labels) in enumerate(tf_dataset):
#     print(f"Batch {batch_idx+1}:")
#     print(f"  Spectrograms shape: {spectrograms.shape}")
#     print(f"  Labels shape: {labels.shape}")
#     if batch_idx == 0:
#         plt.figure(figsize=(10, 4))
#         plt.imshow(tf.transpose(spectrograms[0]).numpy(), aspect='auto', origin='lower')
#         plt.title('Augmented Mel Spectrogram Example (TensorFlow)')
#         plt.colorbar()
#         plt.tight_layout()
#         plt.show()
#     if batch_idx == 1:
#         break

print("\nCode execution complete. Remember to replace dummy data with your actual BirdCLEF+ dataset paths and labels.")




