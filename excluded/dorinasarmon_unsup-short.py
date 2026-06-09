#Data handling
import os
import pandas as pd
import itertools
from PIL import Image

# Randomization
import random

# Audio handling
!pip install PySoundFile
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
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Resizing

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Initialize random number generation
random_seed = 42
random.seed(random_seed)
rng = np.random.default_rng()

# NN training parameters
# target_shape=(256, 512)
segment_length=3
overlap=0.5
latent_dim=256
batch_size = 32
lr = 1e-3
patience = 10
epochs = 50
num_files=1000 # number of files used for training
num_clusters=24

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
print(f"Number of labeled parts (true positive): {num_tp_rows}")


# Chooses files randomly from the given folder
def random_files(source_path, num_files=1):

    all_files=os.listdir(source_path)
    chosen_files=random.sample(all_files, num_files)
    return [os.path.join(source_path, f) for f in chosen_files]


random_file, sr=librosa.core.load(random_files(train_path)[0])
print(sr)


# Returns random audio segment
def random_audio_segment(file_path, segment_length=3.0, sr=22050):
    
    y, sr = librosa.load(file_path, sr=sr)
    total_length=librosa.get_duration(y=y, sr=sr)

    # Random start point
    start_time=random.uniform(0, total_length-segment_length)
    segment_samples=int(segment_length*sr)
    start_sample=int(start_time*sr)
    end_sample=start_sample+segment_samples
    return y[start_sample:end_sample], sr


def spec_gen_short(file_path, sr=22050, n_mels=128, segment_length=3):
    
    audio, sr=random_audio_segment(file_path, segment_length, sr)
    S=librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_db=librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    return S_norm


def spec_save_short(source_path, save_path, num_files=100, sr=22050, segment_length=3, n_mels=128):

    files=[f for f in os.scandir(source_path) if f.is_file()]
    files=files[:num_files]
    for f in tqdm(files):
        if f.is_file():
            output=os.path.join(save_path, os.path.splitext(f.name)[0]+".png")
            spec=spec_gen_short(f.path, sr=sr, segment_length=segment_length)
            spec=(spec*255).astype(np.uint8)
            spec=Image.fromarray(spec, mode='L')
            spec.save(output)


# Generates spectrogram from the given file
def spec_gen(file_path, target_shape=(128, 256), sr=22050):
    
    audio, sr=librosa.core.load(file_path)
    S=librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=target_shape[0]) # f_max, f_min 
    S_db=librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1]>target_shape[1]:
        S_db=S_db[:, :target_shape[1]]
    elif S_db.shape[1]<target_shape[1]:
        pad_width=[(0, 0), (0, target_shape[1]-S_db.shape[1])]
        S_db=np.pad(S_db, pad_width=pad_width, mode='constant')
    
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    return S_norm


# Generates and saves spectrograms from a given number of files
def spec_save(source_path, save_path, target_shape=(128, 256), num_files=100, sr=22050):
    
    files=[f for f in os.scandir(source_path) if f.is_file()]
    files=files[:num_files]
    for f in tqdm(files):
        if f.is_file():
            output=os.path.join(save_path, os.path.splitext(f.name)[0]+".png")
            spec=spec_gen(f.path, target_shape, sr)
            spec=(spec*255).astype(np.uint8)
            spec=Image.fromarray(spec, mode='L')
            spec.save(output)


# spec_save(train_path, save_path, target_shape, num_files=num_files, sr=sr)
spec_save_short(train_path, save_path, num_files, sr, segment_length, n_mels=128)


def conv_autoencoder(input_shape=(128, 256, 1), latent_dim=64):

    # Encoder
    encoder_input=layers.Input(shape=input_shape)
    # x=layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoder_input)
    # x=layers.MaxPooling2D((2,2), padding='same')(x)
    x=layers.Conv2D(64, (3,3), activation='relu', padding='same')(encoder_input)
    x=layers.MaxPooling2D((2,2), padding='same')(x)
    x=layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x=layers.MaxPooling2D((2,2), padding='same')(x)
    x=layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x=layers.MaxPooling2D((2,2), padding='same')(x)
    x_shape=x.shape[1:]
    x_prod=np.prod(x_shape)

    # Bottleneck
    x=layers.Flatten()(x)
    latent=layers.Dense(latent_dim, activation='relu')(x)
    encoder=models.Model(encoder_input, latent)

    # Decoder
    decoder_input=layers.Input(shape=(latent_dim,))
    x=layers.Dense(x_prod, activation='relu')(decoder_input)
    x=layers.Reshape((x_shape))(x)

    # x = layers.Conv2DTranspose(128, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same')(x)
    
    # x=layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    # x=layers.UpSampling2D((2, 2))(x)
    # x=layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    # x=layers.UpSampling2D((2, 2))(x)
    # x=layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # x=layers.UpSampling2D((2, 2))(x)
    # x=layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    # x=layers.UpSampling2D((2, 2))(x)
    decoder_output=layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder_output=Resizing(input_shape[0], input_shape[1])(decoder_output)
    decoder=models.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_output=decoder(encoder(encoder_input))
    autoencoder=models.Model(encoder_input, autoencoder_output)

    return autoencoder, encoder, decoder


# Optimizer
optimizer=keras.optimizers.Adam(learning_rate=lr)

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = patience / 2, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)


# Calculate input shape
example_path=os.path.join(save_path, os.listdir(save_path)[0])
example_image=Image.open(example_path).convert('L')
example_array=np.array(example_image)
target_shape=example_array.shape
print(f"Target shape: {target_shape}")


autoencoder, encoder, decoder=conv_autoencoder(input_shape=(target_shape[0], target_shape[1], 1), latent_dim=latent_dim)
autoencoder.compile(optimizer=optimizer, loss='mse')
autoencoder.summary()


def load_images(source_path, target_shape=(128, 256)):

    images=[]
    for f in os.listdir(source_path):
        if f.endswith('.png'):
            image_path=os.path.join(source_path, f)
            image=Image.open(image_path).convert('L')
            image=image.resize(target_shape)
            image_array=np.array(image)/255.0
            images.append(image_array)
    return np.array(images)


images=load_images(save_path, target_shape=target_shape).reshape(-1, target_shape[0], target_shape[1], 1)


print(images.shape)


history=autoencoder.fit(
    images,
    images,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


def feature_extraction(spectrograms, encoder, n_clusters=24):
    
    X_features=spectrograms.reshape(-1, target_shape[0], target_shape[1], 1)
    latent_features=encoder.predict(X_features, verbose=0)
    print(f"Shape of latent_features: {latent_features.shape}")
    normalized_latent_features=StandardScaler().fit_transform(latent_features)
    kmeans=KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_seed,
        verbose=1
    )
    cluster_labels=kmeans.fit_predict(normalized_latent_features)
    return cluster_labels, latent_features, kmeans


cluster_labels, latent_features, kmeans=feature_extraction(images, encoder, num_clusters)
print(f"Found {len(np.unique(cluster_labels))} clusters.")

cluster_counts=np.bincount(cluster_labels)
print("Number of files in each cluster:")
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}:\t{count}")


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
    unique_clusters=np.unique(cluster_labels)
    for i, c in enumerate(unique_clusters):
        plt.subplot(4, 6, i+1)
        cluster_indices=np.where(cluster_labels==c)[0]
        if len(cluster_indices)>0:
            plt.imshow(images[cluster_indices[0]], aspect='auto', origin='lower')
            plt.title(f'Cluster {c}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


visualization(images, cluster_labels, latent_features)


def visualize_latent_features(decoder, latent_dim, n_cols=8):

    n_rows=int(np.ceil(latent_dim/n_cols))
    plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(latent_dim):
        latent_vector=np.zeros((1, latent_dim))
        latent_vector[0, i]=1.0
        latent_img=decoder.predict(latent_vector)
        latent_img=latent_img.squeeze()
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(latent_img, cmap='gray')
        plt.title(f"Feature {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


visualize_latent_features(decoder, latent_dim, 8)


def segment_audio(file_path, segment_duration=3.0, overlap=0.5):
    
    # audio, sr = librosa.load(file_path)
    # segment_length = int(segment_duration * sr)
    # hop_length = int(segment_length * (1 - overlap))
    
    # segments = []
    # starts = []
    # for start in range(0, len(audio) - segment_length + 1, hop_length):
    #     end = start + segment_length
    #     segment = audio[start:end]
    #     segments.append(segment)
    #     starts.append(start)
    
    # return np.array(segments), starts, sr   


    audio, sr = librosa.load(file_path, sr=sr)
    segment_length = int(segment_duration * sr)
    segments = []
    
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    
    return segments, sr


def segments_to_specs(segments, sr=22050):
    
    specs = []
    for seg in segments:
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
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


# def combine_specs_with_overlap(rec_specs, overlap=0.5)



def combine_specs(specs):
    return np.concatenate(specs, axis=1)


def reconstruct_full_spectrogram(file_path, autoencoder, segment_length=3.0, overlap=0.5):
    
    segments, starts, sr= segment_audio(file_path, segment_length, overlap)
    specs = segments_to_specs(segments, sr)
    rec_specs, rec_h, rec_w = reconstruct_segments(specs, autoencoder)
    # full_rec_spec=combine_specs_with_overlap(rec_specs, overlap)
    full_rec_spec=combine_specs(rec_specs)

    # original=spec_gen(file_path, (rec_h, rec_w), sr)
    y, sr = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    original = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(full_rec_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Reconstructed")
    plt.tight_layout()
    plt.show()

    print(original.shape)
    print(full_rec_spec.shape)
    
    return full_rec_spec



def spec_reconstruct(file_path, autoencoder, target_shape=(128, 256)):

    S_norm=spec_gen(file_path, target_shape)
    input_image=S_norm.reshape(1, target_shape[0], target_shape[1], 1)
    reconstructed_image=autoencoder.predict(input_image, verbose=1)[0, :, :, 0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(S_norm, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Reconstructed')
    plt.tight_layout()
    plt.show()
    print(target_shape)


example_file=random_files(train_path)[0]
# spec_reconstruct(example_file, autoencoder, target_shape)
rec_spec = reconstruct_full_spectrogram(example_file, autoencoder, segment_length, overlap)
Audio(example_file, rate=sr)

