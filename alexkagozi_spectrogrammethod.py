# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # "jax" or "tensorflow" or "torch"

import keras_cv
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

import librosa
import IPython.display as ipd
import librosa.display as lid

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as ply
import ipywidgets as widgets
import seaborn as sns

from itertools import cycle
# Set interactive backend
%matplotlib inline
cmap = mpl.cm.get_cmap('coolwarm')
sns.set_theme(style="white", palette=None)
color_pal = ply.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(ply.rcParams["axes.prop_cycle"].by_key()["color"])

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


DATASET_PATH = '/kaggle/input/birdclef-2024'


class_names = sorted(os.listdir(f"{DATASET_PATH}/train_audio/"))
num_classes = len(class_names)
class_labels = list(range(num_classes))
label2name = dict(zip(class_labels, class_names))
name2label = {v:k for k,v in label2name.items()}


## Print out the first 5 items in the label2name and name2label dictionaries
print(f"Number of classes: {num_classes}")
print({k: label2name[k] for k in list(label2name)[:5]})
print({k: name2label[k] for k in list(name2label)[:5]})


df = pd.read_csv(f'{DATASET_PATH}/train_metadata.csv')
df['filepath'] = DATASET_PATH + '/train_audio/' + df.filename
df['target'] = df.primary_label.map(name2label)
df['filename'] = df.filepath.map(lambda x: x.split('/')[-1])
df['xc_id'] = df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

## display a few rows of the dataframe from columns ['scientific_name', 'scientific_name',  'filepath']
df = df.sample(frac=1, random_state=42)
df.head(5)


## Display the number of samples per class and save the result in a dictionary
class_counts = df.primary_label.value_counts()
class_counts = class_counts.sort_index()
class_counts
# ## Save to a csv file
pd.DataFrame(class_counts.items(), columns=['class', 'count']).to_csv('class_counts.csv', index=False)

## Show the largest and smallest classes
class_counts_csv = pd.read_csv('class_counts.csv')
## Show the largest and smallest classes with the corresponding counts
# Find the minimum and maximum counts
min_count = class_counts.min()
max_count = class_counts.max()
 
print(f"Smallest class: {class_counts_csv['class'][class_counts_csv['count'].idxmin()]} {min_count}")
print(f"Largest class: {class_counts_csv['class'][class_counts_csv['count'].idxmax()]} {max_count}")


## Explore more statistics of the dataset
statistics = class_counts_csv['count'].describe(percentiles=[.25, .5, .75])

# Show statistics
print(f"Mean: {statistics['mean']}")
print(f"Median (50%): {statistics['50%']}")
print(f"Standard Deviation: {statistics['std']}")
print(f"Minimum: {statistics['min']}")
print(f"Maximum: {statistics['max']}")
print(f"25th Percentile: {statistics['25%']}")
print(f"50th Percentile (Median): {statistics['50%']}")
print(f"75th Percentile: {statistics['75%']}")

# You can also calculate specific quantiles, for example:
quantile_10 = class_counts_csv['count'].quantile(0.10)
quantile_90 = class_counts_csv['count'].quantile(0.90)

print(f"10th Percentile: {quantile_10}")
print(f"90th Percentile: {quantile_90}")



percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
percentile_counts = {p: (class_counts_csv['count'] < class_counts_csv['count'].quantile(p)).sum() for p in percentiles}
print("Number of classes below each percentile:")
for p, count in percentile_counts.items():
    print(f"{int(p*100)}%: {count} classes")


## Number of classes with less than 14 samples
less_than_20 = (class_counts_csv['count'] < 20).sum()
print(f"Number of classes with less than 20 samples: {less_than_20}")


## Plot the distribution of class sizes
plt.figure(figsize=(12, 6))
plt.bar(class_counts.index, class_counts.values, color=color_pal)
plt.xticks(rotation=90)
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Distribution of class sizes')
plt.show()



### Group the classes by the class counts and plot the distribution of class sizes
# Group the classes by the class counts
class_counts = pd.DataFrame(class_counts)
class_counts['class'] = class_counts.index
class_counts['count'] = class_counts['count'].astype(int)
class_counts['group'] = pd.cut(class_counts['count'], bins=[0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], labels=['0-10', '10-20', '20-50', '50-100', '100-200', '200-500', '500-1000', '1000-2000', '2000-5000'])
# Plot the distribution of class sizes
plt.figure(figsize=(12, 6))
plt.bar(class_counts['group'].cat.codes, class_counts['count'], color=color_pal)
plt.xticks(rotation=90)
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Distribution of class sizes')
plt.show()


## Check the size of the dataframe
## The shape of the dataframe should be (24459, 15) which means there are 24459 rows and 15 columns
print(f"Dataframe shape: {df.shape}")


## Drop classes with less than 20 samples
# 1. Load class names and metadata
class_names = sorted(os.listdir(f"{DATASET_PATH}/train_audio/"))
df = pd.read_csv(f'{DATASET_PATH}/train_metadata.csv')
df['filepath'] = DATASET_PATH + '/train_audio/' + df.filename
df['filename'] = df.filepath.map(lambda x: x.split('/')[-1])
df['xc_id'] = df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
df['primary_label'] = df['primary_label'].astype(str)  # Ensure consistent type

# 2. Filter out classes with fewer than 20 samples
label_counts = df['primary_label'].value_counts()
valid_labels = label_counts[label_counts >= 20].index.tolist()
df = df[df['primary_label'].isin(valid_labels)]

# 3. Recompute class info
class_names = sorted(valid_labels)
num_classes = len(class_names)
class_labels = list(range(num_classes))
label2name = dict(zip(class_labels, class_names))
name2label = {v: k for k, v in label2name.items()}

# 4. Reset the `target` column using updated name2label
df['target'] = df['primary_label'].map(name2label)

# 5. Shuffle and show
df = df.sample(frac=1, random_state=42)
print(f"Number of classes: {num_classes}")
print({k: label2name[k] for k in list(label2name)[:5]})
print({k: name2label[k] for k in list(name2label)[:5]})
df.head(5)



## Load the audio as a waveform `y`
# Store the sampling rate as `sr`
def load_audio(filepath):
    audio, sr = librosa.load(filepath)
    return audio, sr


import random
for i in range(2):
    # random_index = random.randint(0, df.shape[0])
    ipd.Audio(df['filepath'].iloc[i])
    audio, sr = load_audio(df['filepath'].iloc[i])
    plt.figure(figsize=(10, 3))
    pd.Series(audio).plot(figsize=(10, 5),
                    lw=1,
                    # title=f"{df['scientific_name'].iloc[i]}",
                    color=color_pal[0])
    ## Zoomed in sample to view waves better:
    plt.show()



#### Understanding the audio data
for i in range(2):
    # random_index = random.randint(0, df.shape[0])
    ipd.Audio(df['filepath'].iloc[i])
    audio, sr = load_audio(df['filepath'].iloc[i])
    print(f"Audio: {audio}")
    print(f"Shape of the audio: {audio.shape}")
## The audio file is a numpy array. However, the size of the arrays are different hence we need to pad/trim the arrays to make them the same size



"""
 function to preview simple spectrograms in decibels. A decibel is a logarithmic unit that expresses 
 the ratio of two values of a physical quantity, often power or intensity.
 """
def audio_to_spectrogram(audio):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    # ax.set_title(f"{df['scientific_name'].iloc[i]} Audio Spectogram", fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

for i in range(2):
    audio, sr = load_audio(df['filepath'].iloc[i])
    audio_to_spectrogram(audio)


"""
While a regular spectrogram uses a linear frequency scale, 
a Mel spectrogram uses the Mel scale, which is designed to better reflect how humans perceive sound.
"""
def audio_to_melspectrogram(audio, sr):
    S = librosa.feature.melspectrogram(y=audio,
                                   sr=sr,
                                   n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    # print(S_db_mel.shape)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    # ax.set_title('Mel Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

for i in range(2):
    random_index = random.randint(0, df.shape[0])
    audio, sr = load_audio(df['filepath'].iloc[i])
    audio_to_melspectrogram(audio, sr)


# Define the sampling rate of the audio signal (32 kHz)
sample_rate = 32000

# Define the maximum frequency to include in the spectrogram (16 kHz)
fmax = 16000

# Define the minimum frequency to include in the spectrogram (20 Hz)
fmin = 20

# Function to compute the Mel-spectrogram of an audio signal
def get_spectrogram(audio):
    # Compute the Mel-spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,  # Input audio signal
        sr=sample_rate,  # Sampling rate of the audio
        n_mels=256,  # Number of Mel bands (frequency bins)
        n_fft=2048,  # Size of the FFT window (determines frequency resolution)
        hop_length=512,  # Number of samples between successive frames (determines time resolution)
        fmax=fmax,  # Maximum frequency to include in the spectrogram
        fmin=fmin,  # Minimum frequency to include in the spectrogram
    )

    # Convert the power spectrogram to decibel (dB) scale
    # This makes the values more perceptually meaningful
    spec = librosa.power_to_db(spec, ref=1.0)  # ref=1.0 is the reference value for dB calculation

    # Normalize the spectrogram to the range [0, 1]
    min_ = spec.min()  # Minimum value in the spectrogram
    max_ = spec.max()  # Maximum value in the spectrogram
    if max_ != min_:  # Avoid division by zero if the spectrogram is constant
        spec = (spec - min_) / (max_ - min_)  # Normalize using min-max scaling

    # print(spec.shape)
    # Return the normalized Mel-spectrogram
    return spec


duration = 15
audio_len = duration * sample_rate
def display_audio(row):
    caption = f'Id: {row.filename} | Name: {row.common_name} | Sci.Name: {row.scientific_name}'
    
    audio, sr = load_audio(row.filepath)
    audio = audio[:audio_len]
    spec = get_spectrogram(audio)
    
    # Audio output widget
    audio_output = widgets.Output()
    with audio_output:
        display(ipd.Audio(audio, rate=sample_rate))
    
    # Plot output widget
    plot_output = widgets.Output()
    with plot_output:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, tight_layout=True)
        # fig.suptitle(caption)
        
        # Plot waveform
        lid.waveshow(audio, sr=sample_rate, ax=ax[0], color='b')
        
        # Plot spectrogram
        lid.specshow(spec, sr=sample_rate, hop_length=512, n_fft=2048,
                     fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel', 
                     cmap='coolwarm', ax=ax[1])
        
        ax[0].set_xlabel('')
        plt.show()

    # Display side-by-side
    display(widgets.HBox([audio_output, plot_output]))



## Display a few audio samples
for i in range(2):
    display_audio(df.sample(1).iloc[0])


# Image and audio parameters
img_size = [128, 384]  # Spectrogram image size (height, width)
batch_size = 64  # Batch size for training
## The hop length is the number of samples between successive frames. 
# The audio length is divided by the width of the image to determine the hop length
hop_length = audio_len // (img_size[1] - 1)  # What does this do? 
nfft = 2028  # FFT window size for computing the spectrogram

## Explain the parameters


def build_decoder(with_labels=True, dim=1024):
    """
    Builds a function to decode and preprocess audio files into spectrograms.
    
    Parameters:
    - with_labels (bool): Whether to return labels along with spectrograms.
    - dim (int): Target audio length (number of samples).
    
    Returns:
    - Function to decode audio files (with or without labels).
    """
    def get_audio(filepath):
        """Loads and decodes an audio file from a given filepath using librosa."""
        def _load_audio(filepath):
            # Load the audio file using librosa
            audio, _ = librosa.load(filepath.numpy().decode('utf-8'), sr=sample_rate, mono=True)
            return audio.astype(np.float32)  # Ensure the audio is in float32 format

        # Use tf.py_function to wrap the librosa call
        audio = tf.py_function(_load_audio, [filepath], tf.float32)
        audio.set_shape([None])  # Set shape to [None] since the length may vary
        return audio

    def crop_or_pad(audio, target_len, pad_mode="constant"):
        """Ensures the audio is of fixed length by either cropping or padding."""
        audio_len = tf.shape(audio)[0]  # Get current length of audio
        diff_len = abs(target_len - audio_len)  # Difference from target length

        if audio_len < target_len:
            # If audio is shorter, pad it randomly on both sides
            pad1 = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            pad2 = diff_len - pad1
            audio = tf.pad(audio, paddings=[[pad1, pad2]], mode=pad_mode)

        elif audio_len > target_len:
            # If audio is longer, randomly crop a section
            idx = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            audio = audio[idx : (idx + target_len)]

        return tf.reshape(audio, [target_len])  # Ensure fixed shape

    def apply_preproc(spec):
        """Applies standardization and normalization to the spectrogram."""
        # Standardization: Zero mean and unit variance
        mean = tf.math.reduce_mean(spec)
        std = tf.math.reduce_std(spec)
        spec = tf.where(tf.math.equal(std, 0), spec - mean, (spec - mean) / std)

        # Min-Max Normalization: Scale values between 0 and 1
        min_val = tf.math.reduce_min(spec)
        max_val = tf.math.reduce_max(spec)
        spec = tf.where(
            tf.math.equal(max_val - min_val, 0), 
            spec - min_val, 
            (spec - min_val) / (max_val - min_val)
        )

        return spec

    def get_target(target):
        """Converts a label into a one-hot encoded vector."""
        target = tf.reshape(target, [1])  # Reshape to single element tensor
        target = tf.cast(tf.one_hot(target, num_classes), tf.float32)  # One-hot encoding
        return tf.reshape(target, [num_classes])  # Reshape to match the output format

    def decode(path):
        """Processes an audio file into a spectrogram image."""
        # Load and preprocess the audio
        audio = get_audio(path)
        audio = crop_or_pad(audio, dim)  # Ensure fixed length
        
        # Convert audio to a Mel-spectrogram
        spec = keras.layers.MelSpectrogram(
            num_mel_bins=img_size[0],  # Number of Mel frequency bins (height of image)
            fft_length=nfft,  # FFT window size
            sequence_stride=hop_length,  # Step size between spectrogram columns
            sampling_rate=sample_rate,  # Sample rate of audio
        )(audio)

        spec = apply_preproc(spec)  # Apply normalization and standardization
        
        # Convert spectrogram into a 3-channel image (for compatibility with CNNs)
        spec = tf.tile(spec[..., None], [1, 1, 3])  # Repeat values along the last axis
        return tf.reshape(spec, [*img_size, 3])  # Reshape to (height, width, 3)

    def decode_with_labels(path, label):
        """Processes an audio file into a spectrogram and returns it with its label."""
        return decode(path), get_target(label)

    return decode_with_labels if with_labels else decode


def build_augmenter():
    """
    Creates an augmentation pipeline for spectrogram images.
    Uses MixUp, time masking, and frequency masking to improve model generalization.
    
    Returns:
        A function that applies random augmentations to images and labels.
    """

    # Define a list of augmentation techniques to apply
    augmenters = [
        keras_cv.layers.MixUp(alpha=0.4),  # MixUp augmentation for blending two images
        keras_cv.layers.RandomCutout(
            height_factor=(1.0, 1.0), width_factor=(0.06, 0.12)
        ),  # Time-masking: Randomly removes sections along the time axis
        keras_cv.layers.RandomCutout(
            height_factor=(0.06, 0.1), width_factor=(1.0, 1.0)
        ),  # Frequency-masking: Randomly removes sections along the frequency axis
    ]

    def augment(img, label):
        """
        Applies the augmentation pipeline to an image-label pair.

        Args:
            img (tf.Tensor): Input spectrogram image.
            label (tf.Tensor): Corresponding label for the image.

        Returns:
            Augmented image and label.
        """

        # Wrap image and label in a dictionary for compatibility with keras_cv augmenters
        data = {"images": img, "labels": label}

        # Apply augmentations with a 35% probability for each augmenter
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.35:
                data = augmenter(data, training=True)

        # Extract and return augmented image and label
        return data["images"], data["labels"]

    return augment


seed = 42
def build_dataset(
    paths, 
    labels=None, 
    batch_size=32,
    decode_fn=None, 
    augment_fn=None, 
    cache=True,
    augment=False, 
    shuffle=2048
):
    """
    Builds a TensorFlow dataset pipeline for audio processing.

    Args:
        paths (list or tf.Tensor): List of file paths to audio files.
        labels (list or tf.Tensor, optional): Corresponding labels for classification. Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        decode_fn (function, optional): Function to decode audio files. Defaults to None.
        augment_fn (function, optional): Function to apply augmentations. Defaults to None.
        cache (bool, optional): Whether to cache the dataset in memory. Defaults to True.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        shuffle (int or bool, optional): Buffer size for shuffling. Set to False to disable shuffling. Defaults to 2048.

    Returns:
        tf.data.Dataset: Preprocessed dataset ready for training.
    """

    # Use default decoder if none is provided
    if decode_fn is None:
        decode_fn = build_decoder(with_labels=(labels is not None), dim=audio_len)

    # Use default augmentation function if none is provided
    if augment_fn is None:
        augment_fn = build_augmenter()

    # Set automatic tuning for dataset performance optimization
    AUTO = tf.data.experimental.AUTOTUNE

    # Create dataset from file paths (with or without labels)
    slices = (paths,) if labels is None else (paths, labels)
    print(f"Labels: {labels}")
    ds = tf.data.Dataset.from_tensor_slices(slices)

    # Apply decoding function to process audio files
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)

    # Cache dataset in memory to speed up subsequent iterations
    if cache:
        ds = ds.cache()

    # Shuffle dataset if required
    if shuffle:
        opt = tf.data.Options()
        ds = ds.shuffle(shuffle, seed=seed)  # Shuffle with seed for reproducibility
        opt.experimental_deterministic = False  # Improve performance by allowing non-deterministic order
        ds = ds.with_options(opt)

    # Batch dataset with a fixed size, ensuring even batch sizes
    ds = ds.batch(batch_size, drop_remainder=True)

    # Apply augmentation if enabled
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTO)

    # Prefetch data to improve training performance
    ds = ds.prefetch(AUTO)

    return ds



from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp (stratified by original targets)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['target'],  # Use original targets
    random_state=42
)

# Second split: 50% validation, 50% test (stratified by temp_df's targets)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['target'],  # Critical: Use temp_df's targets!
    random_state=42
)

print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")


# Prepare training dataset
train_paths = train_df.filepath.values  # Extract file paths from training DataFrame
train_labels = train_df.target.values   # Extract corresponding labels

train_ds = build_dataset(
    paths=train_paths, 
    labels=train_labels, 
    batch_size=batch_size,
    shuffle=True,  # Enable shuffling for training dataset
    augment=True  # Apply augmentation for training dataset
)

# Prepare validation dataset
valid_paths = valid_df.filepath.values  # Extract file paths from validation DataFrame
valid_labels = valid_df.target.values   # Extract corresponding labels

valid_ds = build_dataset(
    paths=valid_paths, 
    labels=valid_labels, 
    batch_size=batch_size,
    shuffle=False,  # No shuffling for validation to ensure consistency
    augment=False  # No augmentation for validation dataset
)

# Prepare test dataset
test_paths = test_df.filepath.values  # Extract file paths from test DataFrame
test_labels = test_df.target.values   # Extract corresponding labels

test_ds = build_dataset(
    paths=test_paths, 
    labels=test_labels, 
    batch_size=1,
    shuffle=False,  # No shuffling for test to ensure consistency
    augment=False  # No augmentation for test dataset
)


## Show the shape of the spectrogram from train_ds
x_train = next(iter(train_ds))[0]
print(x_train.shape)
## The shape of the spectrogram is (64, 128, 384, 3) which means that there are 64 images in the batch,
## each image has a height of 128 pixels, a width of 384 pixels and 3 channels (RGB)



def plot_batch(batch, row=3, col=3, label2name=None,):
    """Plot one batch data"""
    if isinstance(batch, tuple) or isinstance(batch, list):
        specs, tars = batch
    else:
        specs = batch
        tars = None
    plt.figure(figsize=(col*5, row*3))
    for idx in range(row*col):
        ax = plt.subplot(row, col, idx+1)
        lid.specshow(np.array(specs[idx, ..., 0]), 
                     n_fft=nfft, 
                     hop_length=hop_length, 
                     sr=sample_rate,
                     x_axis='time',
                     y_axis='mel',
                     cmap='coolwarm')
        if tars is not None:
            label = tars[idx].numpy().argmax()
            name = label2name[label]
            plt.title(name)
    plt.tight_layout()
    plt.show()


sample_ds = train_ds.take(10)
batch = next(iter(sample_ds))
plot_batch(batch, label2name=label2name)


# Create an input layer for the model
inp = keras.layers.Input(shape=(None, None, 3))
preset = 'efficientnetv2_b2_imagenet'
# Pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    preset,
)
out = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    name="classifier"
)(inp)
# Build model
model = keras.models.Model(inputs=inp, outputs=out)
# Compile model with optimizer, loss and metrics
model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
                 metrics=[keras.metrics.AUC(name='auc'), 
                       # keras.metrics.CategoricalAccuracy(name='accuracy'), 
                       # keras.metrics.Precision(name='precision'), 
                       # keras.metrics.Recall(name='recall'), 
                       # keras.metrics.F1Score(name='f1_score')
                        ]
             )
model.summary()



import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback


lr_cb = get_lr_callback(batch_size, plot=True)


ckpt_cb = keras.callbacks.ModelCheckpoint("efficientNet-method.weights.h5",
                                         monitor='val_auc',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')


epochs = 10
# history = model.fit(
#      train_ds, 
#     validation_data=valid_ds, 
#     epochs=epochs,
#      callbacks=[
#         lr_cb,
#         ckpt_cb
#     ],
#     verbose=1
# )

# 3. Train with GPU
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        callbacks=[lr_cb, ckpt_cb],
        verbose=1
    )


## Load the saved weights
model.load_weights("efficientNet-method.weights.h5")


# Get prediction probabilities
y_pred_proba = model.predict(test_ds, verbose=1)
# Get predicted class labels
y_pred = y_pred_proba.argmax(axis=1)
# True labels
y_true = test_df.target.values


print(y_true.shape, y_pred.shape, y_pred_proba.shape)


import numpy as np

unique_classes = np.unique(y_true)
print(f"Unique classes in y_true: {len(unique_classes)} out of {y_pred_proba.shape[1]} total classes")

missing_classes = set(range(y_pred_proba.shape[1])) - set(unique_classes)
print(f"Missing classes: {missing_classes}")




# calculate the accuracy score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score,recall_score
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

## Calculate auc_roc score
auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
print(f"AUC ROC: {auc_roc:.4f}")

## calculate the f1 score
f1 = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score: {f1:.4f}")
## calculate the precision score
precision = precision_score(y_true, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
## calculate the recall score
recall = recall_score(y_true, y_pred, average='macro')
print(f"Recall: {recall:.4f}")



## Create another model with a different backbone
inp = keras.layers.Input(shape=(None, None, 3))
preset = 'resnet50_imagenet'
# Pretrained backbone
backbone = keras_cv.models.ResNet50Backbone.from_preset(
    preset,
)
out = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    name="classifier"
)(inp)
# Build model
model = keras.models.Model(inputs=inp, outputs=out)
# Compile model with optimizer, loss and metrics
model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
                 metrics=[keras.metrics.AUC(name='auc')]
             )
model.summary()


lr_cb = get_lr_callback(batch_size, plot=True)


ckpt_cb = keras.callbacks.ModelCheckpoint("resnet50-method.weights.h5",
                                         monitor='val_auc',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')


## Train the model for 10 epochs
epochs = 10
# history = model.fit(
#      train_ds, 
#     validation_data=valid_ds, 
#     epochs=epochs,
#     callbacks=[lr_cb, ckpt_cb], 
#     verbose=1
# )
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        callbacks=[lr_cb, ckpt_cb],
        verbose=1
    )


## Load the resnet model weights
model.load_weights("resnet50-method.weights.h5")


# Get prediction probabilities
y_pred_proba = model.predict(test_ds, verbose=1)
# Get predicted class labels
y_pred = y_pred_proba.argmax(axis=1)
# True labels
y_true = test_df.target.values



# calculate the accuracy score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score,recall_score
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

## Calculate auc_roc score
auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
print(f"AUC ROC: {auc_roc:.4f}")

## calculate the f1 score
f1 = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score: {f1:.4f}")
## calculate the precision score
precision = precision_score(y_true, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
## calculate the recall score
recall = recall_score(y_true, y_pred, average='macro')
print(f"Recall: {recall:.4f}")





