#pip install opencv-python



#pip install geopandas


#pip install folium


import os
import seaborn as sns

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
import librosa
import random
import time

import cv2
import matplotlib.pyplot as plt
from glob import glob
import geopandas as gpd
from shapely.geometry import Point
import folium
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path


# === PATHS & DATA ===
DATA_DIR = Path("C:\\Users\\Asus\\Desktop\\birdclef-2025")
train_audio = DATA_DIR / "train_audio"
train_soundscapes = DATA_DIR / "train_soundscapes"
test_soundscapes = DATA_DIR / "test_soundscapes"
train_df= pd.read_csv(DATA_DIR / "train.csv")
taxonomy_df = pd.read_csv(DATA_DIR / "taxonomy.csv")
SAMPLE_SUB = pd.read_csv(DATA_DIR / "sample_submission.csv")


from pathlib import Path

# Map all available files with full paths
def index_all_audio_files(train_audio_path):
    audio_index = {}
    for f in Path(train_audio_path).rglob("*.ogg"):
        audio_index[f.name] = f
    return audio_index



taxonomy_df.columns


taxonomy_df.info()


from pathlib import Path

train_audio_dir = Path("/kaggle/input/birdclef-2025/train_audio")
ogg_files = list(train_audio_dir.rglob("*.ogg"))

print(f"Total .ogg files found: {len(ogg_files)}")



# Count number of clips per species (primary label)
species_counts = train_df['primary_label'].value_counts().reset_index()
species_counts.columns = ['primary_label', 'num_clips']
print(species_counts.head())



# Merge with taxonomy to get class_name
species_info = species_counts.merge(taxonomy_df[['primary_label', 'common_name', 'class_name']], on='primary_label')
print(species_info.head())



bird_df = species_info[species_info['class_name'] == 'Aves']
print(f"Total bird species: {len(bird_df)}")



import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
bird_df.sort_values('num_clips', ascending=False).head(30).plot(
    x='common_name', y='num_clips', kind='bar', legend=False, color='skyblue')
plt.ylabel("Number of Audio Clips")
plt.title("Top 30 Bird Species by Clip Count")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()



print("First 5 train_df filenames:")
print(train_df['filename'].head().tolist())

print("\nFirst 5 audio_index keys:")
raw_index = index_all_audio_files(train_audio)
audio_index = {k.lower().strip(): v for k, v in raw_index.items()}
print(list(audio_index.keys())[:5])



valid_filenames = set(audio_index.keys())
filtered_df = train_df[train_df['filename'].isin(valid_filenames)]
print(f"Filtered from {len(train_df)} to {len(filtered_df)}")



# 2. Constants
SAMPLE_RATE = 32000
DURATION = 5  # in seconds
NUM_CLASSES = len(taxonomy_df)  # total classes in BirdCLEF+ 2025
INPUT_SHAPE = (300, 300, 3)
CONFIDENCE_THRESHOLD = 0.6



def audio_to_spectrogram(file_path):
    file_path = Path(file_path)
    
    # Basic checks
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != '.ogg':
        raise ValueError(f"Unsupported file type (not .ogg): {file_path}")
    
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Audio integrity checks
    if y is None or len(y) < sr * 1:
        raise ValueError(f"Audio too short or empty in {file_path}")
    if np.max(np.abs(y)) < 0.001:
        raise ValueError(f"Audio is mostly silent in {file_path}")

    # Generate spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(melspec)
    img = cv2.resize(logmel, (300, 300))
    img = np.stack([img, img, img], axis=-1)
    return img / 255.0



# === FILE INDEXING ===
def index_all_audio_files(audio_root):
    audio_index = {}
    for f in Path(audio_root).rglob("*.ogg"):
        audio_index[f.name.lower()] = f
    print(f"Indexed {len(audio_index)} audio files.")
    return audio_index


class AudioDataGenerator(Sequence):
    def __init__(self, df, audio_index, taxonomy_df, batch_size=32, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_index = audio_index
        self.taxonomy_df = taxonomy_df
        self.batch_size = batch_size
        self.augment = augment
        print(f"[✓] Initialized AudioDataGenerator with {len(self.df)} samples.")

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = [], []
        for _, row in batch.iterrows():
            try:
                file_path = str(self.audio_index[row['filename']])
                spec = audio_to_spectrogram(file_path)
                if spec.shape != INPUT_SHAPE:
                    raise ValueError("Incorrect spectrogram shape")
                if self.augment:
                    spec = self.augment_spectrogram(spec)
                X.append(spec)
                label = np.zeros(NUM_CLASSES)
                idx_tax = self.taxonomy_df[self.taxonomy_df['primary_label'] == row['primary_label']].index[0]
                label[idx_tax] = 1
                y.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        print(f"[✓] Generated batch {idx + 1}/{self.__len__()} with {len(X)} samples.")
        return np.array(X), np.array(y)

    def augment_spectrogram(self, spec):
        if random.random() < 0.5:
            spec += np.random.normal(0, 0.01, spec.shape)
        if random.random() < 0.5:
            t0 = random.randint(0, spec.shape[1] - 10)
            spec[:, t0:t0 + 10, :] = 0
        if random.random() < 0.5:
            f0 = random.randint(0, spec.shape[0] - 10)
            spec[f0:f0 + 10, :, :] = 0
        if random.random() < 0.5:
            spec = np.roll(spec, random.randint(-20, 20), axis=1)
        if random.random() < 0.5:
            spec = np.roll(spec, random.randint(-5, 5), axis=0)
        if random.random() < 0.3:
            spec += np.random.normal(0, 0.03, spec.shape)  # Stronger noise

        return np.clip(spec, 0, 1)


# === LOAD TRAINING + VALIDATION GENERATORS ===
def load_training_generator(split=0.1):
    train_df['filename'] = train_df['filename'].apply(lambda x: Path(x).name.lower().strip())
    audio_index = {f.name.lower(): f for f in Path(train_audio).rglob("*.ogg")}
    filtered_df = train_df[train_df['filename'].isin(audio_index)].copy()
    train_df_split, val_df_split = train_test_split(filtered_df, test_size=split, stratify=filtered_df['primary_label'], random_state=42)

    return (
        AudioDataGenerator(train_df_split, audio_index, taxonomy_df, augment=True),
        AudioDataGenerator(val_df_split, audio_index, taxonomy_df, augment=False)
    )


# === BUILD MODEL ===
def build_model():
    inputs = Input(shape=INPUT_SHAPE)
    base = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    outputs = Dense(NUM_CLASSES, activation='sigmoid')(x)
    return Model(inputs, outputs)



# === TRAIN MODEL (with callbacks and monitoring) ===
def train_model(model, train_gen, val_gen, X_pseudo, y_pseudo):
    X_val, y_val = val_gen[0]
    X_train, y_train = train_gen[0]
    X_train = np.concatenate([X_train, X_pseudo])
    y_train = np.concatenate([y_train, y_pseudo])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("checkpoint_best_model.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="logs")
    ]

    history = {}
    print("Starting base training...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    hist1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks
    )
    history['pretrain'] = hist1.history

    print("Starting fine-tuning...")
    model.trainable = True
    for layer in model.layers[:100]:  # freeze shallow layers
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    hist2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Let EarlyStopping decide
        callbacks=callbacks
    )
    history['finetune'] = hist2.history


    # Plot confusion matrix on validation data
    y_val_pred = model.predict(X_val)
    y_val_bin = (y_val_pred > 0.5).astype(int)
    cm = confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_val_bin, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return model, history



def generate_pseudo_labels(audio_dir, model, taxonomy_df, threshold=CONFIDENCE_THRESHOLD):
    start_time = time.time()
    pseudo_X, pseudo_y = [], []
    audio_files = list(Path(audio_dir).rglob("*.ogg"))
    print(f"[✓] Found {len(audio_files)} audio files for pseudo-labeling.")

    for file in audio_files:
        try:
            spec = audio_to_spectrogram(str(file))
            if spec.shape != INPUT_SHAPE:
                print(f"[✗] Skipping {file.name}, invalid spectrogram shape: {spec.shape}")
                continue
            prob = model.predict(np.expand_dims(spec, axis=0), verbose=0)[0]
            if np.max(prob) > threshold:
                pseudo_X.append(spec)
                pseudo_y.append(prob)
                print(f"[✓] Pseudo-label from {file.name} with max prob {np.max(prob):.2f}")
            else:
                print(f"[✗] Low confidence ({np.max(prob):.2f}) on {file.name}, skipped.")
        except Exception as e:
            print(f"[!] Pseudo-labeling error for {file.name}: {e}")

    print(f"[✓] Total pseudo-labeled samples: {len(pseudo_X)} in {time.time() - start_time:.2f} seconds.")
    return np.array(pseudo_X), np.array(pseudo_y)



# === GEO FILTERING ===
def build_geo_map():
    geo_map = {}
    for _, row in train_df.iterrows():
        geo_map[row['filename']] = (row['latitude'], row['longitude'])
    print(f"Built geo_map with {len(geo_map)} entries.")
    return geo_map

def apply_geo_filtering(preds, df, geo_map, threshold=0.5):
    filtered_preds = preds.copy()
    count = 0
    for i, row in df.iterrows():
        latlon = geo_map.get(row['filename'])
        if latlon:
            if latlon[0] < -10:  # dummy filtering condition
                filtered_preds[i] = (preds[i] > threshold) * preds[i]
                count += 1
    print(f"Applied geo filtering to {count} entries.")
    return filtered_preds


import folium

# === TRAINING HISTORY ===
def plot_training_history(history):
    for phase in history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history[phase]['accuracy'], label='train')
        plt.plot(history[phase]['val_accuracy'], label='val')
        plt.title(f'{phase} accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history[phase]['loss'], label='train')
        plt.plot(history[phase]['val_loss'], label='val')
        plt.title(f'{phase} loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"training_curve_{phase}.png")


# === EVALUATION ===
def evaluate_predictions(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_true, y_pred_bin))
    print("F1 (macro):", f1_score(y_true, y_pred_bin, average='macro'))
    print("F1 (micro):", f1_score(y_true, y_pred_bin, average='micro'))
    print("Precision:", precision_score(y_true, y_pred_bin, average='macro'))
    print("Recall:", recall_score(y_true, y_pred_bin, average='macro'))


# === SUBMISSION ===
def create_submission(preds, row_ids, geo_filtered=False):
    df = pd.DataFrame(preds, columns=[f"c{i}" for i in range(preds.shape[1])])
    df.insert(0, "row_id", row_ids)
    suffix = "geo" if geo_filtered else "raw"
    df.to_csv(f"submission_{suffix}.csv", index=False)



# === TEST TIME AUGMENTATION (TTA) ===
def predict_with_tta(model, X, n=3):
    print(f"Running TTA with {n} augmentations...")
    preds = np.zeros((len(X), NUM_CLASSES))
    for i in range(n):
        X_aug = np.array([np.clip(x + np.random.normal(0, 0.01, x.shape), 0, 1) for x in X])
        preds += model.predict(X_aug, verbose=0)
        print(f"TTA round {i + 1}/{n} completed.")
    return preds / n


# === MAP ===
def plot_predictions_on_map(preds, metadata, threshold=0.5):
    fmap = folium.Map(location=[0, -60], zoom_start=3)
    for i, row in metadata.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if pd.notnull(lat) and pd.notnull(lon):
            ids = np.where(preds[i] > threshold)[0]
            if len(ids):
                label = ", ".join([f"Species {sid}" for sid in ids])
                folium.Marker(location=[lat, lon], popup=label).add_to(fmap)
    fmap.save("prediction_map.html")


def run_pipeline():
    print("Loading training data...")
    train_gen, val_gen = load_training_generator()

    print("Building model...")
    model = build_model()

    print("Generating pseudo-labels...")
    X_pseudo, y_pseudo = generate_pseudo_labels(
        audio_dir=DATA_DIR / "train_soundscapes",
        model=model,
        taxonomy_df=taxonomy_df
    )

    print("Training model with pseudo-labeled data...")
    model, history = train_model(model, train_gen, val_gen, X_pseudo, y_pseudo)

    model.save("efficientnetb3_finetuned.h5")
    print("✅ Saved model as efficientnetb3_finetuned.h5")

    print("Plotting training history...")
    plot_training_history(history)

    print("Evaluating on validation set...")
    X_val, y_val = val_gen[0]
    y_val_pred = model.predict(X_val)
    evaluate_predictions(y_val, y_val_pred)

    print("Predicting test soundscapes...")
    test_files = list((DATA_DIR / "test_soundscapes").glob("*.ogg"))
    X_test = []

    for f in test_files:
        try:
            if not f.exists():
                print(f"[✗] File does not exist: {f}")
                continue
            if f.suffix.lower() != ".ogg":
                print(f"[✗] Skipping non-ogg file: {f}")
                continue
            spec = audio_to_spectrogram(f)
            if spec.shape != INPUT_SHAPE:
                print(f"[✗] Invalid spectrogram shape {spec.shape} for file: {f.name}")
                continue
            X_test.append(spec)
        except Exception as e:
            print(f"[✗] Error processing {f.name}: {e}")

    if len(X_test) == 0:
        print("[✗] No valid test spectrograms were created. Please check your test files.")
        return

    X_test = np.array(X_test)
    row_ids = [f"soundscape_{f.stem}_{i*5}" for f in test_files for i in range(12)]

    print("Running TTA predictions...")
    preds = predict_with_tta(model, X_test)
    create_submission(preds, row_ids, geo_filtered=False)

    print("Applying geo filtering...")
    geo_map = build_geo_map()
    preds_geo = apply_geo_filtering(preds.copy(), train_df.iloc[:len(preds)], geo_map)
    create_submission(preds_geo, row_ids, geo_filtered=True)

    plot_predictions_on_map(preds_geo, train_df.iloc[:len(preds_geo)])



# Run
if __name__ == "__main__":
    run_pipeline()

