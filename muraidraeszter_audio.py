#Data handling
import os
import pandas as pd
from PIL import Image

# Audio handling
!pip install PySoundFile
import librosa
from IPython.display import Audio

# Visualization
import matplotlib.pyplot as plt

# Feedback with progress bar
from tqdm.notebook import tqdm

# Math & Algorithms
import numpy as np


# Initialise random number generation
random_seed = 42
rng = np.random.default_rng()

# NN training parameters
slice_length = 5
sr = None # Use recording native sr
batch_size = 16
lr = 1e-3
epochs = 30
patience = 10

# Folder for storing generated spectrograms
spect_save_path='/kaggle/working/spectrograms'
os.makedirs(spect_save_path, exist_ok=True)


# Root data path for RainForest Species
input_path='/kaggle/input/rfcx-species-audio-detection'

# Train and Test audio recordings data
train_path=os.path.join(input_path, 'train')
test_path=os.path.join(input_path, 'test')

# Labels
tp_label_csv_path=os.path.join(input_path, 'train_tp.csv')


df_labels=pd.read_csv(tp_label_csv_path)
df_labels['t_length'] = df_labels['t_max'] - df_labels['t_min']
lengths = df_labels['t_length']
print(f"Length of labeled audio segments: {lengths.mean():.2f}±{lengths.std():.2f} ({lengths.min():.2f}-{lengths.max():.2f}) [s]")
df_labels


recording_id = None
#recording_id = '5b5218aba'


if recording_id == None:
    # Find all .flac files
    flac_files=[f for f in os.listdir(train_path) if f.endswith('.flac')]
    print(f'No. of .flac samples: {len(flac_files)}')

    # Select and access a random .flac from the list
    file=rng.choice(flac_files)
    file_path=os.path.join(train_path, file)
    recording_id=file.replace('.flac', '')
else:
    file = recording_id + '.flac'
    file_path=os.path.join(train_path, file)

print(f'Selected sample: {file}')

# Read labels and audio data
record = df_labels.loc[df_labels['recording_id'] == recording_id]
audio, rec_sr = librosa.core.load(file_path, sr = sr, mono=False)
print(f"Sample rate: {rec_sr}")

# Print label info
print(f"\nLabels:")
print(f"-------------")
if len(record)==0:
    print("No associated label")
else:
    for _, row in record.iterrows():
        print(f"Faj: {row['species_id']}")
        print(f"Típus: {row['songtype_id']}")
        print(f"Időtartam: {row['t_min']} - {row['t_max']}")
        print(f"Frekvencia: {row['f_min']} - {row['f_max']}\n")

# Display player
Audio(audio, rate=rec_sr)


# Generate the Spectrogram
S = librosa.feature.melspectrogram(y=audio, sr=rec_sr)
S_db = librosa.power_to_db(S, ref=np.max)

#Display
fig, ax = plt.subplots(figsize=(16, 4))
image = librosa.display.specshow(S_db, sr=rec_sr, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(image, ax=ax)
ax.set(title='Mel-Spectrogram')
fig.show()


# Generate spectrogram
S = librosa.feature.melspectrogram(y=audio[0:int(slice_length*rec_sr)], sr=rec_sr)
S_db = librosa.power_to_db(S, ref=np.max)

# Convert to image apropriate format
S_norm = (S_db-S_db.min())/(S_db.max()-S_db.min())
S_norm = (S_norm*255).astype(np.uint8)

# Convert to PIL image
img = Image.fromarray(S_norm)

# Save the image with PIL
img = Image.fromarray(S_norm)
img.save('/kaggle/working/test.png')

# Load the image
img_loaded = Image.open('/kaggle/working/test.png')
S_norm_loaded = np.array(img_loaded)

# Assert equality of saved and loaded array
try:
    np.testing.assert_array_equal(S_norm, S_norm_loaded)
except AssertionError:
    print(f"The saved and loaded arrays are NOT identical:")
else:
    print(f"The saved and loaded arrays are IDENTICAL:")
finally:
    print(f"\tSaved array: {S_norm.shape}; saved values: {S_norm.min()}-{S_norm.max()}; format: {type(S_norm)}")
    display(img)
    print(f"\tLoaded array: {S_norm_loaded.shape}; saved values: {S_norm_loaded.min()}-{S_norm_loaded.max()}; format: {type(S_norm_loaded)}")
    display(img_loaded)


def spectrogram_gen(audio_folder_path, recording_id, species_id,
                    time_min, time_max, sr, slice_length,
                    spect_save_path):
    """Generates spectrograms of from a given audio file

    Generates spectrograms of 'slice_length' between 'time_min' and 'time_max'
    of the given recording.

    Parameters
    ----------
    audio_folder_path : path str
        The folder in which the audio files are stored
    recording_id : str
        The identifier of the audio recording
    species_id : int
        The identifier of the species in the record
    time_min: float
        Start time of the vocalization of the species within the recording
    time_max: float
        End time of the vocalization of the species within the recording
    sr : int
        Forced sample_rate for laoding the audio (None uses the recording native sr)
    slice_length: int
        Length of recording slice to turn into spectrograms in seconds
    spect_save_path: path str
        Folder path to save generated spectrograms into

    Returns
    -------
    save_path: path str
        Location at which the generated spectrogram is saved at
    """

    # Load the audio
    file_path = os.path.join(audio_folder_path, recording_id + '.flac')
    audio, rec_sr = librosa.core.load(file_path, sr=sr, mono=True)

    # Generate audio slice(s)
    slice_time = time_max - time_min
    noSlices = max(int(np.round(slice_time/slice_length)), 1) # How many slices can fit into the given intervall, rounded to nearest int
    
    for i in range(noSlices): 
        # Find center time of the given slice
        center = (time_min + i * (slice_time / (noSlices+1)))

        # Find start and end sample of the given slice
        start = int(max(center - slice_length/2, 0) * rec_sr)
        end = start + int(slice_length * rec_sr)
        if end > len(audio):
            end = len(audio)
            start = end - int(slice_length * rec_sr)

        # Get the sliced audio
        sliced_audio=audio[start:end]

        # Generate Spectrogram
        S = librosa.feature.melspectrogram(y = sliced_audio, sr=rec_sr)
        S_db=librosa.power_to_db(S, ref=np.max)
        
        S_norm=(S_db-S_db.min())/(S_db.max()-S_db.min())
        S_norm = (S_norm*255).astype(np.uint8)
        spect_size = S_norm.shape

        # Save the array as an image
        species_path=os.path.join(spect_save_path, str(species_id))
        os.makedirs(species_path, exist_ok=True)

        filename = f'{species_id}_{recording_id}_{center:.2f}.png' # {center} kell, hátha ugyanolyan nevű file keletkezne
        save_path = os.path.join(species_path, filename)

        S_image = Image.fromarray(S_norm)
        S_image.save(save_path)
    
    return spect_size, save_path # későbbi visszanézésre

# Spectrogram generation progress (with TQDM progress bar)
input_size = None
for i in tqdm(range(len(df_labels))):
    row = df_labels.iloc[i]
    
    recording_id=row['recording_id']
    species_id=row['species_id']
    time_min=float(row['t_min'])
    time_max=float(row['t_max'])

    # Generate spectrogram
    spect_size, save_path = spectrogram_gen(audio_folder_path = train_path, recording_id = recording_id, species_id = species_id,
                                            time_min = time_min, time_max = time_max, sr = sr, slice_length = slice_length, 
                                            spect_save_path = spect_save_path)
    if input_size == None:
        input_size = spect_size
    else:
        if (input_size != spect_size):
            print(f"WARNING: spectrogram size for label {i} ({spect_size}) does not match the spectrogram size for the first label ({input_size})")


species = [int(f) for f in os.listdir(spect_save_path) if os.path.isdir(os.path.join(spect_save_path, f))]
species.sort()
numSpecies = len(species)
print(f"Fajok száma: {numSpecies}")

sum_files=0
print("Fájlok száma az egyes species mappákban:")
for f in species:
    path = os.path.join(spect_save_path, str(f))
    numFiles = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    sum_files += numFiles
    print(f"{f}:\t{numFiles}")
print(f"Összes spectrogram: {sum_files}")


# Imports
import keras
from keras import layers
from keras.metrics import BinaryAccuracy
import tensorflow as tf
import random
from glob import glob
from sklearn.model_selection import train_test_split


# Seeds
os.environ['PYTHONHASHSEED'] = str(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Hiding warnings
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2 -> warnings


all_files = glob("/kaggle/working/spectrograms/*/*.png")
print(f"Total files: {len(all_files)}")
train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=random_seed)

# Input size
image_size = np.array(Image.open(all_files[0]))
input_shape = (image_size.shape[0], image_size.shape[1], 1) # adding grayscale dimension
print(f"Image size: {image_size}\nInput shape: {input_shape}")


# Data augmentation
data_augmentation=keras.Sequential([
    layers.RandomTranslation(
        height_factor=0,
        width_factor=0.1,
        fill_mode='nearest'
    ),
    layers.RandomContrast(0.2),
    layers.RandomZoom(
        height_factor=0,
        width_factor=0.1
    )
])


class AudioDataset(keras.utils.Sequence):
    
    
    # Initialization
    def __init__(self, files, num_species, batch_size=16, shuffle=False, seed=None, **kwargs):
        """ Spectrogram dataset loader - reproducible
            
            Arguments:
            files: list of paths to images
            numSpecies: number of labels
            batch_size: number of files in a batch
            shuffle: whether to shuffle images
            seed: seed for reproducibility
        """
        super().__init__()      

        self.files=files.copy()
        self.num_species=num_species
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.seed=seed
        self.rng=np.random.RandomState(seed) if seed is not None else np.random
        
        # setting seeds
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)

        # detecting image shape
        with Image.open(self.files[0]) as image:
            self.input_shape=(image.size[1], image.size[0], 1)

        self.end_of_epoch()


    # Number of batches
    def __len__(self):
        return int(np.ceil(len(self.files)/self.batch_size))


    # Creates a single batch
    def __getitem__(self, index):
        batch_files=self.files[index*self.batch_size:(index+1)*self.batch_size]

        batch_images=[]
        batch_labels=[]
        
        for f in batch_files:
            with Image.open(f) as image:
                image=image.resize(self.input_shape[:2][::-1])
                image=np.array(image, dtype=np.float32)/255.0
                if image.ndim==2:
                    image=image[...,np.newaxis]
            batch_images.append(image)

            label=int(os.path.basename(os.path.dirname(f)))
            one_hot=np.zeros(self.num_species, dtype=np.float32)
            one_hot[label]=1.0
            batch_labels.append(one_hot)

        return np.stack(batch_images), np.stack(batch_labels)

    
    # Shuffles files
    def end_of_epoch(self):
        if self.shuffle:
            if self.seed is not None:
                permutation=self.rng.permutation(len(self.files))
                self.files=[self.files[i] for i in permutation]
            else:
                np.random.shuffle(self.files)
        

    # Shape of an image
    def image_shape(self):
        return self.input_shape


# Dataset loading
train_dataset=AudioDataset(files=train_files, num_species=numSpecies, batch_size=batch_size, shuffle=True, seed=random_seed)
val_dataset=AudioDataset(files=val_files, num_species=numSpecies, batch_size=batch_size, shuffle=False, seed=random_seed)


# Model
model_keras=keras.models.Sequential([
    
    layers.Input(shape=train_dataset.image_shape()),
    data_augmentation,
    
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(numSpecies, activation='sigmoid')
])


# Optimizer
optimizer=keras.optimizers.Adam(learning_rate = lr)

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = patience / 2, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)


# Compile model
model_keras.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[BinaryAccuracy()]
)
model_keras.summary()


history_keras = model_keras.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stop, reduce_lr])


# Generate spectrograms from a given file
def gen_test_spectrograms(file_path, sr, length, target_shape=(input_shape[0], input_shape[1])):
    spectrograms=[]
    audio, rec_sr = librosa.core.load(file_path, sr=sr, mono=True)
    slice_length = int(rec_sr * length)
    n = len(audio) // slice_length

    for i in range(n):
        start = i * slice_length
        end = start + slice_length
        if end > len(audio):
            end = len(audio)
        sliced_audio = audio[start:end]

        S = librosa.feature.melspectrogram(y=sliced_audio, sr=rec_sr, n_mels=target_shape[0])
        S_db = librosa.power_to_db(S, ref=np.max)      

        if S_db.shape[1]>target_shape[1]:
            S_db=S_db[:, :target_shape[1]]
        elif S_db.shape[1]<target_shape[1]:
            pad_width=[(0, 0), (0, target_shape[1]-S_db.shape[1])]
            S_db=np.pad(S_db, pad_width=pad_width, mode='constant')

        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        spectrograms.append(S_norm[..., None])  # Channel dimension

    return spectrograms
        


# Prediction on test files
def predict_test(model, spectrograms, threshold=0.5):
    inputs=np.array(spectrograms)
    outputs=model.predict(inputs, verbose=0)
    pred=np.max(outputs, axis=0)
    binary_pred=(pred>threshold).astype(int)
    return pred, binary_pred


# Creating .csv file for submission
def create_csv(model, test_path, csv_file=None):
    rows = []
    test_paths = os.listdir(test_path)

    for i in tqdm(range(len(test_paths))):
        file = test_paths[i]
        
        if file.endswith('.flac'):
            file_path = os.path.join(test_path, file)
            recording_id = file.replace('.flac', '')
            spectrograms = gen_test_spectrograms(file_path, sr = sr, length = slice_length)
            pred, _ = predict_test(model_keras, spectrograms)
            rows.append([recording_id] + list(pred))
            # _, binary_pred=predict_test(model_keras, spectrograms)
            # rows.append([recording_id]+list(binary_pred))

    df = pd.DataFrame(rows, columns=['recording_id']+[f"s{i}" for i in range(numSpecies)])
    if csv_file:
        df.to_csv(csv_file, float_format='%.5f', index=False)
    else:
        print(df)


# Saving submission file
submission_dir = '/kaggle/working/csv'
os.makedirs(submission_dir, exist_ok=True)
csv_file = os.path.join(submission_dir, 'rainForest_submission_keras.csv')
create_csv(model_keras, test_path, csv_file=csv_file)


# num_files=500
# #patience=5


# def load_audio_files(folder, rng, num_files=None, seed=42):
#     all_files=glob(os.path.join(folder, "*.flac"))
#     if num_files:
#         all_files=rng.choice(all_files, size=num_files, replace=False, shuffle=False).tolist()
#     return all_files


# def load_audio(file_path, sr=22050):
#     audio, _=librosa.load(file_path, sr=sr)
#     return audio


# def gen_slices(
#     audio,
#     sr,
#     slice_length=3,
# ):
#     num_samples=int(slice_length*sr)
#     num_slices=len(audio)//num_samples
#     specs=[]
#     for i in range(num_slices):
#         start=i*num_samples
#         end=start+num_samples
#         s=audio[start:end]
#         S=librosa.feature.melspectrogram(y=s, sr=sr, n_mels=128)
#         S_db=librosa.power_to_db(S, ref=np.max)
#         S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
#         specs.append(S_norm.astype("float32"))

#     return specs


# def gen_spectrogram_files(
#     model,
#     train_path,
#     rng,
#     num_files=None,
#     slice_length=3,
#     sr=22050,
#     threshold=0.5,
#     seed=42
# ):
#     files=load_audio_files(train_path, rng, num_files, seed)
#     all_spectrograms=[]
#     all_labels=[]
    
#     for f in tqdm(files):
#         audio=load_audio(f, sr)
#         slices=gen_slices(audio, sr, slice_length)
#         if not slices:
#             continue
#         inputs=np.array(slices)[..., np.newaxis] # channel dimension
#         outputs_prob=model.predict(inputs, verbose=0)
#         outputs_pred=(outputs_prob>threshold).astype(np.float32)
        
#         all_spectrograms.extend(inputs)
#         all_labels.extend(outputs_pred)

#     specs=np.array(all_spectrograms, dtype=np.float32)
#     labels=np.array(all_labels, dtype=np.float32)

#     return specs, labels


# inputs, outputs=gen_spectrogram_files(model_keras, train_path, rng, num_files=num_files, slice_length=slice_length, sr=rec_sr, seed=random_seed)


# model_keras.compile(
#     optimizer=optimizer,
#     loss="binary_crossentropy",
#     metrics=[BinaryAccuracy()]
# )


# history_new=model_keras.fit(
#     inputs,
#     outputs,
#     batch_size=16,
#     validation_split=0.1,
#     epochs=epochs,
#     callbacks=[early_stop, reduce_lr]
# )


# def gen_submission(
#     submission_dir,
#     model,
#     test_path,
#     filename
# ):
#     os.makedirs(submission_dir, exist_ok=True)
#     csv_file=os.path.join(submission_dir, filename)
#     create_csv(model, test_path, csv_file=csv_file)
#     return csv_file


# submission_dir = '/kaggle/working/csv'
# filename='rainForest_submission_keras_new.csv'
# gen_submission(submission_dir, model_keras, test_path, filename)

