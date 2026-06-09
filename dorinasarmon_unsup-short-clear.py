#Data handling
import os
import pandas as pd
# import itertools
from PIL import Image

# Randomization
import random

# Audio handling
# !pip install PySoundFile
import librosa
from IPython.display import Audio

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
segment_length = 1
latent_dim = 256
batch_size = 32
lr = 1e-3
patience = 10
epochs = 100
num_files = 1000
num_clusters = 24

# Initialize random number generation
random_seed = 42
random.seed(random_seed)
rng = np.random.default_rng()


# Folder for storing generated spectrograms
save_path='/kaggle/working/spectrograms'
os.makedirs(save_path, exist_ok=True)

# Root data path for RainForest Species
input_path='/kaggle/input/rfcx-species-audio-detection'

# Train and Test audio recordings data
train_path=os.path.join(input_path, 'train')
test_path=os.path.join(input_path, 'test')

# Labels
tp_label_csv_path=os.path.join(input_path, 'train_tp.csv')


# Number of files
num_train_files=len([f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))])
num_test_files=len([f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))])
num_tp_rows=len(pd.read_csv(tp_label_csv_path))
print(f"Number of training files: {num_train_files}")
print(f"Number of test files: {num_test_files}")
print(f"Number of labeled entries (true positive): {num_tp_rows}")


# Chooses files randomly from the given folder
def random_files(source_path, num_files=1):

    all_files=os.listdir(source_path)
    chosen_files=random.sample(all_files, num_files)
    return [os.path.join(source_path, f) for f in chosen_files]


# Returns random audio segment
def random_audio_segment(file_path, segment_length=3.0, sr = None):
    
    y, sr = librosa.load(file_path, sr = sr)
    total_length = librosa.get_duration(y=y, sr = sr)

    # Random start point
    start_time = random.uniform(0, total_length-segment_length)
    segment_samples = int(segment_length * sr)
    start_sample = int(start_time * sr)
    end_sample = start_sample+segment_samples
    return y[start_sample:end_sample], sr


# Generates spectrogram from the given file
def spec_gen(file_path, sr = None):
    """
    Generates a spectogram from a given audio file, and returns the first
    target_shape[1] pixel columns from it.
    """
    
    audio, sr = librosa.core.load(file_path)
    S = librosa.feature.melspectrogram(y = audio, sr = sr, n_mels = 128)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1]>target_shape[1]:
        S_db=S_db[:, :target_shape[1]]
    elif S_db.shape[1]<target_shape[1]:
        pad_width=[(0, 0), (0, target_shape[1]-S_db.shape[1])]
        S_db=np.pad(S_db, pad_width=pad_width, mode='constant')
    
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    return S_norm


def spec_gen_short(file_path, sr = None, n_mels = 128, segment_length = 3):
    
    audio, sr = random_audio_segment(file_path, segment_length, sr)
    S = librosa.feature.melspectrogram(y = audio, sr = sr, n_mels = n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    return S_norm


def spec_save_short(source_path, save_path, num_files = 100, sr = None, segment_length = 3, n_mels = 128):
    """
    Saves a random segment from the first n audio files from the
    source path as a .png image to save_path.
    """

    files = [f for f in os.scandir(source_path) if f.is_file()]
    files = files[:num_files]
    for f in tqdm(files):
        output = os.path.join(save_path, os.path.splitext(f.name)[0]+".png")
        spec = spec_gen_short(f.path, sr = sr, segment_length=segment_length)
        spec = (spec*255).astype(np.uint8)
        # spec = Image.fromarray(spec, mode = 'L')
        spec = Image.fromarray(spec)
        spec.save(output)


def load_images(source_path):

    images=[]
    for f in os.listdir(source_path):
        if f.endswith('.png'):
            image_path = os.path.join(source_path, f)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)/255.0
            images.append(image_array)
    return np.array(images)


class AudioDataset(keras.utils.Sequence):

    # Initialization
    def __init__(self, images, batch_size, shuffle=False, seed=None, **kwargs):

        super().__init__()

        self.images=images
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
        self.input_shape=self.images[0].shape
        self.indices=np.arange(len(self.images))

        self.end_of_epoch()


    # Number of batches
    def __len__(self):
        return int(np.ceil(len(self.images)/self.batch_size))


    # Creates a single batch
    def __getitem__(self, index):
        batch_idx=self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_images=[self.images[i] for i in batch_idx]
        batch_images=np.stack(batch_images).astype("float32")

        if batch_images.ndim==3:
            batch_images=np.expand_dims(batch_images, -1)

        if batch_images.max()>1.0:
            batch_images=batch_images/255.0

        return batch_images, batch_images


    # Shuffles files
    def end_of_epoch(self):
        if self.shuffle:
            if self.seed is not None:
                self.rng.shuffle(self.indices)
            else:
                np.random.shuffle(self.indices)


    # Shape of an image
    def image_shape(self):
        return self.input_shape


# Getting sampling rate
random_file, sr = librosa.core.load(random_files(train_path)[0], sr = None)
print(sr)

# Saving a short spec segment from each file
spec_save_short(train_path, save_path, num_files, segment_length = segment_length, n_mels=128)


# Calculating input shape
example_path=os.path.join(save_path, os.listdir(save_path)[0])
example_image=Image.open(example_path).convert('L')
example_array=np.array(example_image)
target_shape=example_array.shape
print(f"Example image path: {example_path}")
print(f"Target shape: {target_shape}")

# Showing example image
plt.imshow(example_image, cmap='viridis')

# Loading and reshaping images
images=load_images(save_path).reshape(-1, target_shape[0], target_shape[1], 1)
print(images.shape)
plt.imshow(images[0], cmap='viridis')

# Building the training dataset for the autoencoder
train_dataset = AudioDataset(
    images=images,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)


def conv_autoencoder(input_shape, latent_dim=64):

    # Encoder
    encoder_input=layers.Input(shape=input_shape)
    x=layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_input)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D((2, 2), padding='same')(x)
    x=layers.Dropout(0.2)(x)
    
    x=layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D((2, 2), padding='same')(x)
    x=layers.Dropout(0.2)(x)
    
    x=layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=layers.BatchNormalization()(x)
    
    x_shape=x.shape[1:]
    x_prod=np.prod(x_shape)

    # Bottleneck
    x=layers.Flatten()(x)
    latent=layers.Dense(latent_dim, activation='sigmoid')(x)
    encoder=models.Model(encoder_input, latent)

    # Decoder
    decoder_input=layers.Input(shape=(latent_dim,))
    x=layers.Dense(x_prod, activation='relu')(decoder_input)
    x=layers.Reshape((x_shape))(x)

    x=layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x=layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x=layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    decoder_output=layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder_output=Resizing(input_shape[0], input_shape[1])(decoder_output)
    decoder=models.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_output=decoder(encoder(encoder_input))
    autoencoder=models.Model(encoder_input, autoencoder_output)

    return autoencoder, encoder, decoder


# Optimizer
optimizer=keras.optimizers.Adam(learning_rate = lr)

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = patience / 2, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)


autoencoder, encoder, decoder=conv_autoencoder(input_shape=(target_shape[0], target_shape[1], 1), latent_dim=latent_dim)
autoencoder.compile(optimizer=optimizer, loss='mse')
autoencoder.summary()
encoder.summary()
decoder.summary()


history = autoencoder.fit(
    train_dataset,
    # images,
    # images,
    epochs=epochs,
    # batch_size = batch_size,
    # validation_split = 0.2,
    callbacks = [early_stop, reduce_lr],
    verbose=1
)


def visualize_latent_features(decoder, latent_dim, n_cols=8):

    # n_rows=int(np.ceil(latent_dim/n_cols))
    # plt.figure(figsize=(n_cols*2, n_rows*2))
    # for i in range(latent_dim):
    #     latent_vector=np.zeros((1, latent_dim))
    #     latent_vector[0, i] = 10
    #     latent_img = decoder.predict(latent_vector)
    #     latent_img = latent_img.squeeze()
    #     plt.subplot(n_rows, n_cols, i+1)
    #     plt.imshow(latent_img, cmap ='viridis')
    #     plt.title(f"Feature {i+1}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    n_rows=int(np.ceil(latent_dim/n_cols))
    latent_matrix = np.zeros((latent_dim, latent_dim), dtype=np.float32)
    for i in range(latent_dim):
        latent_matrix[i, i] = 10
    decoded = decoder.predict(latent_matrix)
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(latent_dim):
        img = decoded[i].squeeze()
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap='viridis')
        plt.title(f"Feature {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(decoded.shape)


# Visualizing latent space
visualize_latent_features(decoder, latent_dim, 8)


# Testing all-zero image
dark = np.zeros([1, target_shape[0], target_shape[1], 1])
latent_dark = encoder.predict(dark)
rec_dark = decoder.predict(latent_dark)
zero_in = decoder.predict(np.zeros([1, latent_dim]))
print(latent_dark)
# print(zero_in)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(dark[0, :, :, 0], cmap='viridis')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(rec_dark[0, :, :, 0], cmap='viridis')
plt.title('Reconstructed')
plt.subplot(1, 3, 3)
plt.imshow(zero_in[0, :, :, 0], cmap='viridis')
plt.title('Zero latent vector')
plt.tight_layout()
plt.show()


def feature_extraction(spectrograms, encoder, n_clusters=24):
    
    X_features = spectrograms.reshape(-1, target_shape[0], target_shape[1], 1)
    latent_features = encoder.predict(X_features, verbose=0)
    print(f"Shape of latent_features: {latent_features.shape}")
    normalized_latent_features = StandardScaler().fit_transform(latent_features)
    kmeans=KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_seed,
        verbose=1
    )
    cluster_labels=kmeans.fit_predict(normalized_latent_features)
    return cluster_labels, latent_features, kmeans


# K-means clustering
cluster_labels, latent_features, kmeans=feature_extraction(images, encoder, num_clusters)
print(f"Found {len(np.unique(cluster_labels))} clusters.")

# Printing cluster numbers
cluster_counts=np.bincount(cluster_labels)
print("Number of files in each cluster:")
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}:\t{count}")


def find_process_files(consistent_df, train_path):
    
    spectrograms = []
    valid_recording_ids = []
    valid_species_labels = []

    # Get list of .flac files
    flac_files = [f for f in os.listdir(train_path) if f.endswith('.flac')]
    print(f"Number of .flac files: {len(flac_files)}")

    # Mapping
    flac_mapping = {}
    for flac_file in flac_files:
        recording_id = os.path.splitext(flac_file)[0]
        flac_mapping[recording_id] = os.path.join(train_path, flac_file)

    # Processing files
    for idx, row in tqdm(consistent_df.iterrows(), total=len(consistent_df), desc="Processing files"):
        recording_id = row['recording_id']
        species_id = row['species_id']
        
        if recording_id in flac_mapping:
            flac_path = flac_mapping[recording_id]
            spectrogram = spec_gen(flac_path)
            spectrograms.append(spectrogram)
            valid_recording_ids.append(recording_id)
            valid_species_labels.append(species_id)
        else:
            print(f".flac file not found for: {recording_id}")

    spectrograms_array = np.array(spectrograms)
    print(f"Generated spectrograms: {len(spectrograms_array)}")
    print(f"Spectrogram shape: {spectrograms_array.shape}")
    
    return spectrograms_array, valid_recording_ids, valid_species_labels


def true_label_vis(recording_id, cluster_labels, true_labels, latent_features):

    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
    X_tsne = tsne.fit_transform(latent_features)

    # Create plot
    colors=plt.cm.tab20.colors+plt.cm.tab10.colors[:4]
    cmap=plt.matplotlib.colors.ListedColormap(colors)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                        c=cluster_labels, 
                        cmap=cmap,
                        alpha=0.7,
                        linewidth=0.5)
    for i, (x, y) in enumerate(X_tsne):
        plt.annotate(str(species_labels[i]), 
                    (x, y), 
                    fontsize=8,
                    fontweight='bold',
                    alpha=0.8)    
    plt.colorbar(scatter, label='Clusters')
    plt.title('t-SNE visualization')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')

    plt.tight_layout()
    plt.show()
    
    return X_tsne


# Getting true positive recording ids
df=pd.read_csv(tp_label_csv_path)
multi_species=df.groupby('recording_id')['species_id'].nunique()
conflicting_ids = multi_species[multi_species > 1].index
same_species_ids = multi_species[multi_species == 1].index

print(f"\nRecording_ids with conflicting species: {len(conflicting_ids)}")
print(f"Recording_ids with consistent species: {len(same_species_ids)}")

# Consistent labeling only
consistent_df = df[df['recording_id'].isin(same_species_ids)].drop_duplicates(subset=['recording_id'])
print(f"Consistent recording_ids after processing: {len(consistent_df)}")
print(consistent_df)

spectrograms, recording_ids, species_labels = find_process_files(consistent_df, train_path)


cluster_labels, latent_features, kmeans = feature_extraction(spectrograms, encoder, num_clusters)
print(f"Found {len(np.unique(cluster_labels))} clusters.")

X_tsne = true_label_vis(recording_ids, cluster_labels, species_labels, latent_features)
cluster_counts = np.bincount(cluster_labels)
print("\nNumber of files in each cluster:")
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}:\t{count} files")


def visualization(images, cluster_labels, latent_features):

    # t-SNE visualization
    tsne=TSNE(n_components=2, perplexity=30, random_state=random_seed, verbose=1)
    features_tsne=tsne.fit_transform(latent_features)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    colors=plt.cm.tab20.colors+plt.cm.tab10.colors[:4]
    cmap=plt.matplotlib.colors.ListedColormap(colors)
    scatter=plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=cluster_labels, cmap=cmap, alpha=0.6)
    plt.colorbar(scatter, ticks=range(num_clusters))
    plt.title('t-SNE visualization of the clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

    # Example spectrogram from each cluster
    unique_clusters = np.unique(cluster_labels)
    for i, c in enumerate(unique_clusters):
        plt.subplot(4, 6, i+1)
        cluster_indices = np.where(cluster_labels==c)[0]
        if len(cluster_indices)>0:
            plt.imshow(images[cluster_indices[0]])
            plt.title(f'Cluster {c}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


# Visualizing cluster labeling
visualization(images, cluster_labels, latent_features)


def segment_audio(file_path, segment_duration=3.0, sr = None):

    audio, sr = librosa.load(file_path, sr = sr)
    segment_length = int(segment_duration * sr)
    segments = []
    
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    
    return segments, audio, sr


def segments_to_specs(segments, sr):
    
    specs = []
    for seg in segments:
        S = librosa.feature.melspectrogram(y=seg, sr = sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        specs.append(S_norm)
    return np.array(specs)


def reconstruct_segments(specs, autoencoder):
    
    reconstructed = []
    for S in specs:
        h, w = S.shape
        input_image = S.reshape(1, h, w, 1)
        rec = autoencoder.predict(input_image, verbose=0)[0, :, :, 0]
        reconstructed.append(rec)
    rec_h, rec_w=reconstructed[0].shape
    return np.array(reconstructed), rec_h, rec_w


def combine_specs(specs):
    return np.concatenate(specs, axis=1)


def reconstruct_full_spectrogram(file_path, autoencoder, segment_length=3.0):
    
    segments, audio, sr = segment_audio(file_path, segment_length)
    specs = segments_to_specs(segments, sr)
    rec_specs, rec_h, rec_w = reconstruct_segments(specs, autoencoder)
    full_rec_spec = combine_specs(rec_specs)

    S = librosa.feature.melspectrogram(y=audio, sr = sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    original = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='viridis', aspect='auto') # aspect nem kell
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(full_rec_spec, cmap='viridis', aspect='auto')
    plt.title("Reconstructed")
    plt.tight_layout()
    plt.show()

    print(original.shape)
    print(full_rec_spec.shape)
    
    return full_rec_spec, audio, sr


# Reconstructing a random segment
def reconstruct_slice(file_path, segment_length, autoencoder, sr=None):
    random_slice, sr=random_audio_segment(file_path, segment_length, sr)
    random_slice_spec=segments_to_specs([random_slice], sr)
    random_slice_rec, rec_h, rec_w=reconstruct_segments(random_slice_spec, autoencoder)
    
    S = librosa.feature.melspectrogram(y=random_slice, sr = sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    original = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='viridis', aspect='auto')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(random_slice_rec[0], cmap='viridis', aspect='auto')
    plt.title("Reconstructed")
    plt.tight_layout()
    plt.show()

    print(original.shape)
    print(random_slice_rec.shape)
    
    return random_slice_rec, random_slice, sr


# Reconstructing example slice
example_file=random_files(train_path)[0]
print(example_file)
rec_spec, audio, sr = reconstruct_slice(example_file, segment_length, autoencoder)
Audio(audio, rate = sr)


# Reconstructing example file
example_file=random_files(train_path)[0]
print(example_file)
rec_spec, audio, sr = reconstruct_full_spectrogram(example_file, autoencoder, segment_length)
Audio(audio, rate = sr)

