import os
import cv2 # OpenCV library for image processing and computer vision
import math
import time
import librosa # Audio analysis library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # A library for displaying progress bars for loops

import torch # PyTorch machine learning framework
import warnings
# warnings.filterwarnings("ignore")


class Config:
    # If set to True, it limits the number of samples to be processed in subsequent data processing, speeding up development and testing.
    DEBUG_MODE: bool = False

    OUTPUT_DIR: str = "/kaggle/working"
    DATA_ROOT: str = "/kaggle/input/birdclef-2025"

    # Set the sampling rate of the speech data to 32 kHz.
    FS: int = 32000

    # Mel spectrogram parameters

    # The Fast Fourier Transform (FFT) window size when performing the Sort Time Fourier Transform(STFT).
    # The larget the window size, the higher the frequency resolution, but the lower the time resolution.
    N_FFT: int = 1024

    # The STFT window hop size (movement).
    # The smaller the value, the higher the time resolution, but the higher the computational cost.
    HOP_LENGTH: int = 512

    # The number of mel filter banks in the mel spectrogram.
    # It will be the height (number of frequency bins) of the generated mel spectrogram.
    N_MELS: int = 128

    # The lowest frequency used in the mel spectrogram calculation (unit: Hz).
    FMIN: int = 50

    # The highest frequency used in the mel spectrogram calculation (unit: Hz).
    FMAX: int = 14000

    # The target length in seconds for processing audio data.
    # Based on this value, the audio data is trimmed.
    TARGET_DURATION: float = 5.0

    # The target shape (height, width) of the generated mel-spectrogram image.
    # After audio processing, the image will be reized to this shape.
    TARGET_SHAPE: tuple[int, int] = (256, 256)

    # The maximum number of samples to be processed.
    N_MAX: int | None = 50 if DEBUG_MODE else None

config: Config = Config()


print(f"Debug mode: {config.DEBUG_MODE}")
print(f"Max samples to process: {config.N_MAX if config.N_MAX is not None else 'ALL'}")


# Load taxonomy (information about biological classification systems) data
taxonomy_df: pd.DataFrame = pd.read_csv(f"{config.DATA_ROOT}/taxonomy.csv")
species_class_map: dict[str, str] = dict(zip(taxonomy_df["primary_label"], taxonomy_df["class_name"]))


taxonomy_df.head()


# Load training metadata
train_df: pd.DataFrame = pd.read_csv(f"{config.DATA_ROOT}/train.csv")


train_df.head()


# Create a label list and mapping dictionary
label_list: list[str] = sorted(train_df["primary_label"].unique())
label_id_list: list[int] = list(range(len(label_list)))
label2id: dict[str, int] = dict(zip(label_list, label_id_list))
id2label: dict[int, str] = dict(zip(label_id_list, label_list))

print(f"Found {len(label_list)} unique species")


# Create a dataframe for preprcessing
working_df: pd.DataFrame = train_df[["primary_label", "rating", "filename"]].copy()


working_df["target"] = working_df.primary_label.map(label2id)


working_df["filepath"] = config.DATA_ROOT + "/train_audio/" + working_df.filename


working_df["samplename"] = working_df.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])


working_df["class"] = working_df.primary_label.map(lambda x: species_class_map.get(x, "Unknown"))


working_df.head()


total_samples: int = min(len(working_df), config.N_MAX or len(working_df))


print(f"Total samples to process: {total_samples} out of {len(working_df)} available")


# Sample by class
print(working_df["class"].value_counts())


def audio2melspec(audio_data: np.ndarray) -> np.ndarray:
    """
    Converts audio data to a normalized Mel spectrogram.

    This function takes a NumPy array representing audio data, calculates its
    Mel spectrogram, converts it to the decibel scale, and normalizes the
    values to the range [0, 1]. It handles potential NaN values in the input
    audio data by filling them with the mean of the non-NaN values.

    Args:
        audio_data (np.ndarray): A NumPy array containing the audio time series data.
                                 Expected to be a 1D array of floating-point numbers.

    Returns:
        np.ndarray: A 2D NumPy array representing the normalized Mel spectrogram.
                    The shape is (n_mels, time_frames), where n_mels is the
                    number of Mel bands and time_frames depends on the length
                    of the audio data and the hop length. The values are
                    normalized to the range [0, 1].
    """
    
    if np.isnan(audio_data).any():
        mean_signal: float = np.nanmean(audio_data)
        audio_data: np.ndarray = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec: np.ndarray = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    mel_spec_db: np.ndarray = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm: np.ndarray = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm


# Start audio processing
print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")


start_time: float = time.time()
all_bird_data: dict[str, np.float32] = {}
errors: list[tuple[str, str]] = []

for i, row in tqdm(working_df.iterrows(), total=total_samples):
    if config.N_MAX is not None and i >= config.N_MAX:
        break
    
    try:
        audio_data: np.ndarray

        # Load audio data
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)

        # Calculates the target number of samples from the specified target length (in seconds) and sampling rate
        target_samples: int = int(config.TARGET_DURATION * config.FS)

        # If the number of samples of the loaded audio data is shorter than the target number of samples,
        # the audio is repeated to increase its length.
        if len(audio_data) < target_samples:
            n_copy: int = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # The starting index for extracting samples of the target length from the center of the audio data.
        start_idx: int = max(0, int(len(audio_data) / 2 - target_samples / 2))

        # The end index of the range to extract, without going beyond the range even if the end of the data is reached.        
        end_idx: int = min(len(audio_data), start_idx + target_samples)
        
        # Extract the audio data between the calculated start and end indexes.
        center_audio: np.ndarray = audio_data[start_idx:end_idx]

        # If the target number of samples is still not reached after extraction
        # (for example because the original audio data was extremely short),
        # the end of the extracted audio data is padded with 0 until the target number of samples is reached.
        if len(center_audio) < target_samples:
            center_audio: np.ndarray = np.pad(
                center_audio,
                (0, target_samples - len(center_audio)),
                mode="constant"
            )

        # Calculate mel spectrogram
        mel_spec: np.ndarray = audio2melspec(center_audio)

        # Resize the calculated mel spectrogram if it differs from the set target shape.
        if mel_spec.shape != config.TARGET_SHAPE:
            mel_spec: np.ndarray = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        # Store the processed mel spectrograms (NumPy Arrays).
        all_bird_data[row.samplename] = mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {row.filepath}: {e}")
        errors.append((row.filepath, str(e)))

end_time: float = time.time()


print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {len(all_bird_data)} files out of {total_samples} total")
print(f"Failed to process {len(errors)} files")


samples: list = []
displayed_classes = set()


max_samples: int = min(4, len(all_bird_data))


for i, row in working_df.iterrows():
    if i >= (config.N_MAX or len(working_df)):
        break
        
    if row["samplename"] in all_bird_data:
        if config.DEBUG_MODE:
            if row["class"] not in displayed_classes:
                samples.append((row["samplename"], row["class"], row["primary_label"]))
                displayed_classes.add(row["class"])
        else:
            if row["class"] not in displayed_classes:
                samples.append((row["samplename"], row["class"], row["primary_label"]))
                displayed_classes.add(row["class"])
        
        if len(samples) >= max_samples:  
            break


if samples:
    plt.figure(figsize=(16, 12))
    
    for i, (samplename, class_name, species) in enumerate(samples):
        plt.subplot(2, 2, i+1)
        plt.imshow(all_bird_data[samplename], aspect="auto", origin="lower", cmap="viridis")
        plt.title(f"{class_name}: {species}")
        plt.colorbar(format="%+2.0f dB")
    
    plt.tight_layout()
    debug_note = "debug_" if config.DEBUG_MODE else ""
    plt.savefig(f"{debug_note}melspec_examples.png")
    plt.show()


output_path: str = f"{config.OUTPUT_DIR}/birdclef2025_melspec_{int(config.TARGET_DURATION)}sec_{config.TARGET_SHAPE[0]}_{config.TARGET_SHAPE[1]}.npy"


np.save(output_path, all_bird_data)




