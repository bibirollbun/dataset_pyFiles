#Data handling
import os
import pandas as pd
from PIL import Image
import shutil

# Randomization
import random

# Audio handling
import librosa
from IPython.display import Audio
import soundfile as sf

# Visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Feedback with progress bar
from tqdm.notebook import tqdm

# Math & Algorithms
import numpy as np

# Model
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Resizing

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# NN training parameters
segment_length = 0.8
n_mels=128
num_batches=20
num_files = 5000

# Initialize random number generation
random_seed = 42
random.seed(random_seed)
rng = np.random.default_rng()


# Returns None if file cannot be read
def safe_load(path, sr):
    try:
        y, s = sf.read(path)
        if sr is not None and s != sr:
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        return y
    except Exception:
        try:
            y, _ = librosa.load(path, sr=sr)
            return y
        except Exception:
            return None


# Generates spectrogram from the audio - normalized
def create_spec(audio, n_mels):
    S=librosa.power_to_db(librosa.feature.melspectrogram(y=audio, n_mels=n_mels), ref=np.max)
    return (S-S.min())/(S.max()-S.min())


# Returns starting points of segments based on the given parameters
def create_slices(y_duration, segment_length, n_slices):
    max_possible = int(y_duration // segment_length)
    n_slices = min(n_slices, max_possible)
    step=y_duration/n_slices
    starts=[i*step for i in range(n_slices)]
    return starts


# Creates spectrogram from a file
def gen_spec_from_file(file_path, n_slices, n_mels, sr, segment_length=None):
    y=safe_load(file_path, sr)
    if y is None:
        return []
    y_duration=librosa.get_duration(y=y, sr=sr)
    specs=[]

     # Whole file
    if segment_length is None:
        specs.append(create_spec(y, n_mels))
        return specs

    # Segmented
    starts=create_slices(y_duration, segment_length, n_slices)
    for s in starts:
        start_sample=int(sr*s)
        end_sample=start_sample+int(sr*segment_length)
        segment=y[start_sample:end_sample]
        specs.append(create_spec(segment, n_mels))
    return specs


# Generates spectrograms from several files (and optionally saves them)
def gen_specs(source_path, n_files, n_slices, n_mels, sr, segment_length=None, save_path=None, files=None):
    if files is None:
        files = [f for f in os.scandir(source_path) if f.is_file()]
        selected_files=files[:n_files]
    else:
        selected_files=files
    specs=[]
    
    for f in tqdm(selected_files):
        file_specs=gen_spec_from_file(f, n_slices, n_mels, sr, segment_length)
        if len(file_specs)==0:
            continue
        # Saving spectrograms
        if save_path is not None:
            base=os.path.splitext(os.path.basename(f))[0]
            for i, spec in enumerate(file_specs):
                spec=(spec*255).astype(np.uint8)
                spec=Image.fromarray(spec)
                output=f"{base}_{i}.png"
                spec.save(os.path.join(save_path, output))
        else:
            specs.extend(file_specs)
    return specs


# Returns a random file from the given folder
def random_files(source_path, num_files=1):
    all_files=os.listdir(source_path)
    chosen_files=random.sample(all_files, num_files)
    return [os.path.join(source_path, f) for f in chosen_files]


# Folder for storing generated spectrograms
save_path='/kaggle/working/spectrograms'
labeled_path='/kaggle/working/labeled'
pred_path='/kaggle/working/test'
os.makedirs(save_path, exist_ok=True)
os.makedirs(labeled_path, exist_ok=True)
os.makedirs(pred_path, exist_ok=True)

# Root data path for RainForest Species
input_path='/kaggle/input/rfcx-species-audio-detection'

# Train and Test audio recordings data
train_path=os.path.join(input_path, 'train')
test_path=os.path.join(input_path, 'test')

# Labels
tp_label_csv_path=os.path.join(input_path, 'train_tp.csv')


# Number of files
training_files=[f.path for f in os.scandir(train_path) if f.is_file()]
test_files=[f.path for f in os.scandir(test_path) if f.is_file()]
num_train_files=len(training_files)
num_test_files=len(test_files)
num_tp_rows=len(pd.read_csv(tp_label_csv_path))
print(f"Number of training files: {num_train_files}")
print(f"Number of test files: {num_test_files}")
print(f"Number of labeled entries (true positive): {num_tp_rows}")


# Shows an example file
example_file=random_files(train_path)[0]
example_audio, sr=librosa.core.load(example_file, sr=None)
print(f"Sampling rate: {sr}")

op1=gen_spec_from_file(example_file, 1, n_mels, sr)
op2=gen_spec_from_file(example_file, 1, n_mels, sr, 10)
op3=gen_spec_from_file(example_file, 1, n_mels, sr, 3)
op4=gen_spec_from_file(example_file, 1, n_mels, sr, 1)

plt.figure(figsize=(15,10))

plt.subplot(4, 1, 1)
plt.imshow(op1[0], cmap='viridis', aspect='auto')
plt.title('Whole file')

plt.subplot(4, 1, 2)
plt.imshow(op2[0], cmap='viridis', aspect='auto')
plt.title('10 sec')

plt.subplot(4, 1, 3)
plt.imshow(op3[0], cmap='viridis', aspect='auto')
plt.title('3 sec')

plt.subplot(4, 1, 4)
plt.imshow(op4[0], cmap='viridis', aspect='auto')
plt.title('1 sec')

plt.tight_layout()
plt.savefig('/kaggle/working/spec_4.png', bbox_inches='tight')
plt.show()


Audio(example_file, rate=sr)


# Create batches from file paths
def make_batches(files, n_batches):
    total=len(files)
    batch_size=int(np.ceil(total/n_batches))

    batches=[]
    for i in range(0, total, batch_size):
        batches.append(files[i:i+batch_size])
    return batches


batches=make_batches(training_files, num_batches)
for i, batch in enumerate(batches):
    print(i, len(batch))


for i in range(num_batches):
    # Create images
    gen_specs(train_path, num_files, 75, n_mels, sr, segment_length=segment_length, save_path=save_path, files=batches[1])
    # Create zip
    zip_name=f"/kaggle/working/spectrograms_batch_{i}"
    shutil.make_archive(zip_name, 'zip', '/kaggle/working/spectrograms')
    # Delete existing images
    for f in os.listdir(save_path):
        path = os.path.join(save_path, f)
        if os.path.isfile(path):
            os.remove(path)


# Deletes .zip file
# zip_path = "/kaggle/working/spectrograms_batch_0.zip"
# if os.path.exists(zip_path):
#     os.remove(zip_path)


# Showing an example image
example_path=random_files(save_path, 1)[0]
example_image = Image.open(example_path)
plt.figure(figsize=(8, 6))
plt.imshow(example_image, cmap='viridis')
plt.title("Example spectrogram for training")


df=pd.read_csv(tp_label_csv_path)
df['t_length'] = df['t_max'] - df['t_min']
lengths = df['t_length']
print(f"Length of labeled audio segments: {lengths.mean():.2f}±{lengths.std():.2f} ({lengths.min():.2f}-{lengths.max():.2f}) [s]")
df


# Generates and saves spectrograms from a given audio file
def gen_and_save_spec_labeled(audio_folder_path, recording_id, species_id,
                    time_min, time_max, segment_length, n_mels, save_path):

    # Load the audio
    file_path = os.path.join(audio_folder_path, recording_id + '.flac')
    audio, sr = librosa.core.load(file_path, sr=None)

    # Number of segments in time interval
    slice_length=time_max-time_min
    num_segments=max(int(slice_length // segment_length), 1)

    for i in range(num_segments):
        # Find center time of the given segment
        center = (time_min + i * (slice_length / (num_segments+1)))

        # Find start and end sample of the given slice
        start = int(max(center - segment_length/2, 0) * sr)
        end = start + int(segment_length * sr)
        if end > len(audio):
            end = len(audio)
            start = end - int(segment_length * sr)
    
        # Get the sliced audio
        audio_short=audio[start:end]

        # Generate spectrogram
        S = create_spec(audio_short, n_mels)
        S = (S * 255).astype(np.uint8)

        
        # For saving .png
        filename = f'{species_id}_{recording_id}_{center:.2f}.png'
        species_path=os.path.join(save_path, str(species_id))
        os.makedirs(species_path, exist_ok=True)
        save_path_final = os.path.join(species_path, filename)

        # Saving image
        image=Image.fromarray(S)
        image.save(save_path_final)

    return save_path_final, S, S.shape


for i in tqdm(range(len(df))):
    row = df.iloc[i]
    
    rec_id = row["recording_id"]
    species_id = row["species_id"]
    time_min = float(row["t_min"])
    time_max = float(row["t_max"])

    gen_and_save_spec_labeled(train_path, rec_id, species_id, time_min, time_max, segment_length=segment_length,n_mels=n_mels, save_path=labeled_path)


# Create .zip for download
zip_name=f"/kaggle/working/spectrograms_labeled"
shutil.make_archive(zip_name, 'zip', '/kaggle/working/labeled')


# Number of species
species = [int(f) for f in os.listdir(labeled_path) if os.path.isdir(os.path.join(labeled_path, f))]
species.sort()
num_species = len(species)
print(f"Species: {num_species}")

sum_files=0
print("Number of files in each species folder:")
for f in species:
    path = os.path.join(labeled_path, str(f))
    num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    sum_files += num_files
    print(f"{f}:\t{num_files}")
print(f"Spectrograms: {sum_files}")

