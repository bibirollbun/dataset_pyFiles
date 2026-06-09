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


import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(audio_path, target_sr=None, duration=None):
    """Loads an audio file.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int, optional): Target sampling rate. If None, uses the
            original sampling rate. Defaults to None.
        duration (float, optional): Target duration in seconds. If the audio
            is shorter, it will be padded with zeros. If longer, it will be
            truncated. Defaults to None.

    Returns:
        tuple: A tuple containing the audio time series (numpy array) and the
               sampling rate.
    """
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
    """Extracts a Mel spectrogram from an audio signal.

    Args:
        audio (np.ndarray): The audio time series.
        sr (int): The sampling rate of the audio.
        n_fft (int, optional): Length of the FFT window. Defaults to 2048.
        hop_length (int, optional): Number of audio samples between adjacent
            STFT columns. Defaults to 512.
        n_mels (int, optional): Number of Mel bands to generate.
            Defaults to 128.

    Returns:
        np.ndarray: The Mel spectrogram (shape: (n_mels, time)).
    """
    if audio is None:
        return None
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def process_audio_file(audio_path, target_sr=32000, duration=5.0,
                       n_fft=1024, hop_length=512, n_mels=64):
    """Loads an audio file and extracts its Mel spectrogram features.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int, optional): Target sampling rate. Defaults to 32000.
        duration (float, optional): Target duration in seconds. Defaults to 5.0.
        n_fft (int, optional): Length of the FFT window. Defaults to 1024.
        hop_length (int, optional): Number of audio samples between adjacent
            STFT columns. Defaults to 512.
        n_mels (int, optional): Number of Mel bands to generate.
            Defaults to 64.

    Returns:
        np.ndarray or None: The Mel spectrogram features if loading was
                             successful, otherwise None.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr, duration=duration)
    if audio is not None:
        features = extract_mel_spectrogram(audio, sr, n_fft, hop_length, n_mels)
        return features
    return None


def process_directory(audio_dir, output_dir, target_sr=32000, duration=5.0,
                      n_fft=1024, hop_length=512, n_mels=64):
    """Processes all audio files in a directory and saves the Mel spectrograms
    as numpy arrays.

    Args:
        audio_dir (str): Path to the directory containing audio files.
        output_dir (str): Path to the directory where the extracted features
                            will be saved.
        target_sr (int, optional): Target sampling rate for all audio files.
            Defaults to 32000.
        duration (float, optional): Target duration for all audio files in
            seconds. Defaults to 5.0.
        n_fft (int, optional): Length of the FFT window. Defaults to 1024.
        hop_length (int, optional): Number of audio samples between adjacent
            STFT columns. Defaults to 512.
        n_mels (int, optional): Number of Mel bands to generate.
            Defaults to 64.
    """
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
    """Visualizes the Mel spectrogram.

    Args:
        mel_spectrogram_db (np.ndarray): Mel spectrogram in dB scale.
        sr (int): The sampling rate of the audio.
        title (str, optional): Title of the plot. Defaults to "Mel Spectrogram".
    """
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
    """Visualizes MFCCs and Chroma features.

    Args:
        mfccs (np.ndarray): Mel-frequency cepstral coefficients.
        chroma (np.ndarray): Chroma features.
        sr (int): The sampling rate of the audio.
        title (str, optional): Title of the plot. Defaults to "Audio Features".
    """
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
    """Loads an audio file and extracts MFCCs and Chroma features.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int, optional): Target sampling rate. Defaults to None.
        duration (float, optional): Target duration in seconds. Defaults to None.
        n_fft (int, optional): Length of the FFT window. Defaults to 2048.
        hop_length (int, optional): Number of audio samples between adjacent
            STFT columns. Defaults to 512.
        n_mfcc (int, optional): Number of Mel-frequency cepstral coefficients
            to compute. Defaults to 20.
        n_chroma (int, optional): Number of chroma bins to produce. Defaults to 12.

    Returns:
        tuple: A tuple containing:
            - mfccs (np.ndarray): Mel-frequency cepstral coefficients
              (shape: (n_mfcc, time)).
            - chroma (np.ndarray): Chroma features (shape: (n_chroma, time)).
            - sr (int): The sampling rate of the audio.
    """
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

