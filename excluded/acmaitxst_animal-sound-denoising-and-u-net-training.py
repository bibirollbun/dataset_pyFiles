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


import pandas as pd

if os.path.exists(csv_path):
    metadata_df = pd.read_csv(csv_path)
    print("CSV metadata loaded. Number of entries:", len(metadata_df))
else:
    print("CSV file not found. Continuing without metadata.")


import glob

# Collect all .ogg files from any subfolder within train_audio_dir
train_audio_files = glob.glob(os.path.join(train_audio_dir, '**/*.ogg'), recursive=True)
print(f"Found {len(train_audio_files)} .ogg files in train_audio (including subfolders).")


train_soundscape_files = [
    os.path.join(train_soundscapes_dir, f)
    for f in os.listdir(train_soundscapes_dir)
    if f.endswith('.ogg')
]
print(f"Found {len(train_soundscape_files)} .ogg files in train_soundscapes.")


def build_noise_profile_from_files(file_list, sr=None):
    noise_profiles = []
    for nf in file_list:
        audio, file_sr = load_audio_file(nf, sr=sr)
        noise_profiles.append(audio)
    combined_noise = np.concatenate(noise_profiles)
    return combined_noise

noise_profile = build_noise_profile_from_files(train_soundscape_files, sr=44100)
print(f"Combined noise profile length: {len(noise_profile)/44100:.2f} seconds")


# Pick a sample animal sound from train_audio_files
if train_audio_files:
    sample_audio_path = train_audio_files[0]
    audio, sr = load_audio_file(sample_audio_path, sr=44100)
    print(f"Sample audio loaded: {len(audio)/sr:.2f} seconds at {sr} Hz")
    S_db_before = compute_spectrogram(audio, sr)
    save_spectrogram(S_db_before, sr, 'sample_before_noise_reduction')
else:
    print("No sample audio found in train_audio.")


# Apply noise reduction to the sample audio
reduced_audio = nr.reduce_noise(audio_clip=audio, noise_clip=noise_profile, verbose=False)
S_db_after = compute_spectrogram(reduced_audio, sr)
save_spectrogram(S_db_after, sr, 'sample_after_noise_reduction')


# Create a training pair from the sample audio (for demonstration)
noisy_spec, clean_spec = create_training_pair(sample_audio_path, noise_profile, sr=44100)
print("Noisy spectrogram shape:", noisy_spec.shape)
print("Clean spectrogram shape:", clean_spec.shape)


def unet_model(input_shape):
    inputs = tf.keras.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    concat4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D((2, 2))(c4)
    concat5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
    model = models.Model(inputs, outputs)
    return model

input_shape = (256, 256, 1)
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


def data_generator(file_list, noise_profile, sr, target_shape=(256,256)):
    """Generator yielding (noisy_spectrogram, denoised_spectrogram) pairs."""
    for file_path in file_list:
        try:
            noisy_spec, clean_spec = create_training_pair(file_path, noise_profile, sr=sr, target_shape=target_shape)
            yield noisy_spec, clean_spec
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create a dataset from all .ogg files in train_audio (recursively collected)
import tensorflow as tf

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_audio_files, noise_profile, sr=44100, target_shape=(256,256)),
    output_signature=(
        tf.TensorSpec(shape=(256,256,1), dtype=tf.float32),
        tf.TensorSpec(shape=(256,256,1), dtype=tf.float32)
    )
)
batch_size = 4
dataset = dataset.shuffle(buffer_size=50).batch(batch_size)


checkpoint_path = "unet_denoiser_best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')


model.fit(dataset, epochs=5, callbacks=[checkpoint])


model.save('unet_denoiser_final_model.h5')
print("Model saved as unet_denoiser_final_model.h5")

