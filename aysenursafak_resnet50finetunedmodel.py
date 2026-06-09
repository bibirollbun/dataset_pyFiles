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
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB3, ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
import librosa
import random
import time
from tensorflow.keras.losses import BinaryFocalCrossentropy

import cv2
import matplotlib.pyplot as plt
from glob import glob
import geopandas as gpd
from shapely.geometry import Point
import folium
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard



# === PATHS & DATA ===
DATA_DIR = Path("/kaggle/input/birdclef-2025")
#train_audio = DATA_DIR / "train_audio"
#train_soundscapes = DATA_DIR / "train_soundscapes"
test_soundscapes = DATA_DIR / "test_soundscapes"
#train_df= pd.read_csv(DATA_DIR / "train.csv")
taxonomy_df = pd.read_csv(DATA_DIR / "taxonomy.csv")
SAMPLE_SUB = pd.read_csv(DATA_DIR / "sample_submission.csv")


from pathlib import Path

# Map all available files with full paths
def index_all_audio_files(train_audio_path):
    audio_index = {}
    for f in Path(train_audio_path).rglob("*.ogg"):
        audio_index[f.name] = f
    return audio_index



from pathlib import Path

'''train_audio_dir = Path("/kaggle/input/birdclef-2025/train_audio")
ogg_files = list(train_audio_dir.rglob("*.ogg"))

print(f"Total .ogg files found: {len(ogg_files)}")'''



# 2. Constants
SAMPLE_RATE = 32000
DURATION = 5  # in seconds
NUM_CLASSES = len(taxonomy_df)  # total classes in BirdCLEF+ 2025
INPUT_SHAPE = (300, 300, 3)
CONFIDENCE_THRESHOLD = 0.5
#  Advanced switches
USE_PSEUDO_LABELS = True
USE_FOCAL_LOSS = True
THRESHOLD = 0.1


print(f" USE_PSEUDO_LABELS: {USE_PSEUDO_LABELS}")
print(f" USE_FOCAL_LOSS: {USE_FOCAL_LOSS}")
print(f" EVAL THRESHOLD: {THRESHOLD}")
print(f" PSEUDO-LABEL THRESHOLD: {CONFIDENCE_THRESHOLD}")


def audio_to_spectrogram(file_path):
    file_path = Path(file_path)
    
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


'''class AudioDataGenerator(Sequence):
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


    #  Advanced augmentation
    def frequency_mask(self, spec, F=20):
        f = random.randint(0, F)
        f0 = random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :, :] = 0
        return spec

    def time_mask(self, spec, T=20):
        t = random.randint(0, T)
        t0 = random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t, :] = 0
        return spec
    
    
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

        return np.clip(spec, 0, 1)'''


'''# === LOAD TRAINING + VALIDATION GENERATORS ===
def load_training_generator(split=0.1):
    train_df['filename'] = train_df['filename'].apply(lambda x: Path(x).name.lower().strip())
    audio_index = {f.name.lower(): f for f in Path(train_audio).rglob("*.ogg")}
    filtered_df = train_df[train_df['filename'].isin(audio_index)].copy()
    train_df_split, val_df_split = train_test_split(filtered_df, test_size=split, stratify=filtered_df['primary_label'], random_state=42)

    return (
        AudioDataGenerator(train_df_split, audio_index, taxonomy_df, augment=True),
        AudioDataGenerator(val_df_split, audio_index, taxonomy_df, augment=False)
    )'''


'''HYPERPARAMS_GRID = [
    {
        "arch": "EfficientNetB2",
        "dropout_rate": 0.4,
        "dense_units": 512,
        "learning_rate": 1e-4,
        "finetune_depth": 80,
        "finetune_lr": 1e-6
    },
    {
        "arch": "EfficientNetB3",
        "dropout_rate": 0.4,
        "dense_units": 512,
        "learning_rate": 1e-4,
        "finetune_depth": 80,
        "finetune_lr": 1e-6
    },
    {
        "arch": "ResNet50",
        "dropout_rate": 0.3,
        "dense_units": 512,
        "learning_rate": 1e-4,
        "finetune_depth": 80,
        "finetune_lr": 1e-6
    }
]
'''


'''def build_model(config):
    if config['arch'] == 'EfficientNetB2':
        base = EfficientNetB2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    elif config['arch'] == 'EfficientNetB3':
        base = EfficientNetB3(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    elif config['arch'] == 'ResNet50':
        base = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    else:
        raise ValueError("Unsupported architecture")

    inputs = Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['dense_units'], activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model'''


'''def train_model(model, train_gen, val_gen, X_pseudo, y_pseudo, config):
    X_val, y_val = val_gen[0]
    X_train, y_train = train_gen[0]
    if USE_PSEUDO_LABELS and len(X_pseudo) > 0:
        print(f"Adding {len(X_pseudo)} pseudo-labeled samples to training data")
        X_train = np.concatenate([X_train, X_pseudo])
        y_train = np.concatenate([y_train, y_pseudo])

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(f"best_{config['arch'].lower()}.h5", save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    history = {}

    print("Feature extraction phase...")
    loss_fn = BinaryFocalCrossentropy(gamma=2.0,  label_smoothing=0.05) if USE_FOCAL_LOSS else 'binary_crossentropy'
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss=loss_fn, metrics=['accuracy'])
    hist1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )
    history['pretrain'] = hist1.history

    print("\n Starting fine-tuning...")
    model.trainable = True
    for layer in model.layers[:-config['finetune_depth']]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(config['finetune_lr']),loss=loss_fn, metrics=['accuracy'])
    hist2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )
    history['finetune'] = hist2.history

    # === Predict on validation set
    print("\n Predicting on validation set...")
    y_val_pred = model.predict(X_val)

    # --- Plot probability histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y_val_pred.flatten(), bins=50, color='skyblue')
    plt.title('Predicted Probabilities Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.show()

    # === Threshold sweep
    thresholds = [0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3]
    best_thresh = 0.0
    best_f1_macro = 0.0

    print("\n Threshold Tuning Results on Validation Set:")
    for thresh in thresholds:
        y_val_bin = (y_val_pred > thresh).astype(int)
        val_acc = accuracy_score(y_val, y_val_bin)
        f1_macro = f1_score(y_val, y_val_bin, average='macro', zero_division=0)
        f1_micro = f1_score(y_val, y_val_bin, average='micro', zero_division=0)
        precision_macro = precision_score(y_val, y_val_bin, average='macro', zero_division=0)
        recall_macro = recall_score(y_val, y_val_bin, average='macro', zero_division=0)

        print(f"\nThreshold: {thresh:.2f}")
        print(f"  Validation Accuracy = {val_acc:.4f}")
        print(f"  F1 (macro) = {f1_macro:.4f}")
        print(f"  F1 (micro) = {f1_micro:.4f}")
        print(f"  Precision (macro) = {precision_macro:.4f}")
        print(f"  Recall (macro) = {recall_macro:.4f}")

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_thresh = thresh

    print("\n Best Threshold on Validation Set:")
    print(f"  Best Threshold = {best_thresh}")
    print(f"  Best F1 (macro) = {best_f1_macro:.4f}")

    # === Confusion matrix at best threshold
    y_val_bin_best = (y_val_pred > best_thresh).astype(int)
    cm = confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_val_bin_best, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Validation Confusion Matrix (Threshold={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # === Save best threshold for submission use
    with open(f"best_threshold_{config['arch'].lower()}.txt", "w") as f:
        f.write(str(best_thresh))

    return model, history
'''


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



'''import folium

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
        plt.savefig(f"training_curve_{phase}.png")'''


# === EVALUATION ===
def evaluate_predictions(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_true, y_pred_bin))
    print("F1 (macro):", f1_score(y_true, y_pred_bin, average='macro'))
    print("F1 (micro):", f1_score(y_true, y_pred_bin, average='micro'))
    print("Precision:", precision_score(y_true, y_pred_bin, average='macro'))
    print("Recall:", recall_score(y_true, y_pred_bin, average='macro'))


def create_submission(preds, row_ids):
    df = pd.DataFrame(preds, columns=[f"c{i}" for i in range(preds.shape[1])])
    df.insert(0, "row_id", row_ids)
    df.to_csv("submission.csv", index=False)
    print(f" Saved submission.csv")


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


'''def run_pipeline():
    print("\n Starting BirdCLEF training pipeline...")
    
    # === Load training & validation generators
    train_gen, val_gen = load_training_generator()
    
    for config in HYPERPARAMS_GRID:
        print("\n====================================")
        print(f" Starting training for architecture: {config['arch']}")
        print("====================================\n")
        
        # === Build model
        model = build_model(config)
        print(f" Built model: {config['arch']}")
        
        # === Pseudo-labeling
        if USE_PSEUDO_LABELS:
            print(" Generating pseudo-labels...")
            X_pseudo, y_pseudo = generate_pseudo_labels(
                audio_dir=DATA_DIR / "train_soundscapes",
                model=model,
                taxonomy_df=taxonomy_df,
                threshold=CONFIDENCE_THRESHOLD
            )
        else:
            print(" Pseudo-labeling is OFF.")
            X_pseudo = np.zeros((0, *INPUT_SHAPE))
            y_pseudo = np.zeros((0, NUM_CLASSES))
        
        print(f" Pseudo-labeled samples: {len(X_pseudo)}")
        
        # === Train model (with threshold tuning)
        model, history = train_model(
            model,
            train_gen,
            val_gen,
            X_pseudo,
            y_pseudo,
            config
        )
        
        # === Save final model with clear name
        model_filename = f"{config['arch'].lower()}_finetuned.h5"
        model.save(model_filename)
        print(f" Model saved as {model_filename}")
        
        # === Plot training curves
        plot_training_history(history)
        
        # === Final Validation Predictions and Metrics
        X_val, y_val = val_gen[0]
        y_val_pred = model.predict(X_val)
        
        print("\n Final Evaluation on Validation Set:")
        for thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
            y_val_bin = (y_val_pred > thresh).astype(int)
            val_acc = accuracy_score(y_val, y_val_bin)
            f1_macro = f1_score(y_val, y_val_bin, average='macro', zero_division=0)
            f1_micro = f1_score(y_val, y_val_bin, average='micro', zero_division=0)
            precision_macro = precision_score(y_val, y_val_bin, average='macro', zero_division=0)
            recall_macro = recall_score(y_val, y_val_bin, average='macro', zero_division=0)
            print(f"\nThreshold: {thresh:.2f}")
            print(f"  Validation Accuracy = {val_acc:.4f}")
            print(f"  F1 (macro) = {f1_macro:.4f}")
            print(f"  F1 (micro) = {f1_micro:.4f}")
            print(f"  Precision (macro) = {precision_macro:.4f}")
            print(f"  Recall (macro) = {recall_macro:.4f}")
        
        print("\n Completed training for architecture:", config['arch'])
'''


# Run
'''if __name__ == "__main__":
    run_pipeline() '''


import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

def submission_pipeline_tta(model_name, model_dir, tta_rounds=3, batch_size=16, features_dir=None, input_shape=INPUT_SHAPE):
    """
    BirdCLEF2025 Kaggle Submission Pipeline, robust to missing test audio and exceptions.
    - Loads a Keras model.
    - If test audio exists, runs inference with TTA (for local debug/validation).
    - If not, builds submission.csv with zeros or custom logic using sample_submission.csv (for competition submission).
    - Always creates a valid submission.csv, even if exceptions occur.
    - Uses precomputed features if features_dir is given.
    """
    try:
        print("\nSUBMISSION PIPELINE (Batchwise, TTA-ready)")
        print(f"Model: {model_name}")
        print(f"TTA rounds: {tta_rounds}")

        # Load model
        model_path = Path(model_dir) / f"{model_name}.h5"
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")

        # Load threshold
        try:
            threshold_path = Path(model_dir) / f"best_threshold_{model_name.lower()}.txt"
            with open(threshold_path, "r") as f:
                THRESHOLD = float(f.read())
            print(f"Loaded best threshold: {THRESHOLD}")
        except Exception:
            THRESHOLD = 0.05
            print(f"Threshold file not found. Using fallback THRESHOLD={THRESHOLD}")

        # Try to find test audio files (for local debug)
        print("Searching for test_soundscapes...")
        test_soundscapes_dir = Path("/kaggle/input/birdclef-2025/test_soundscapes")
        test_files = list(test_soundscapes_dir.rglob("*.ogg")) if test_soundscapes_dir.exists() else []

        if len(test_files) > 0:
            # LOCAL DEBUG PIPELINE: process test audio with TTA
            print(f"Found {len(test_files)} test .ogg files")
            # (Implement your audio_to_spectrogram and create_submission as needed.)
            all_preds = []
            all_row_ids = []

            def yield_batches(files, batch_size):
                batch = []
                meta = []
                for f in files:
                    try:
                        spec = audio_to_spectrogram(f)  # You must implement this function!
                        if spec.shape != input_shape:
                            print(f"[✗] Invalid spectrogram for {f.name}")
                            continue
                        batch.append(spec)
                        meta.append(f)
                        if len(batch) == batch_size:
                            yield batch, meta
                            batch = []
                            meta = []
                    except Exception as e:
                        print(f"[✗] Error processing {f.name}: {e}")
                if batch:
                    yield batch, meta

            print("Starting batch processing + TTA...")
            for batch, metas in yield_batches(test_files, batch_size):
                batch = np.array(batch)
                preds = np.zeros((len(batch), model.output_shape[1]))
                for i in range(tta_rounds):
                    X_aug = np.clip(batch + np.random.normal(0, 0.01, batch.shape), 0, 1)
                    preds += model.predict(X_aug, verbose=0)
                preds /= tta_rounds
                preds_bin = (preds > THRESHOLD).astype(int)
                all_preds.extend(preds_bin)
                for f in metas:
                    all_row_ids.extend([f"soundscape_{f.stem}_{i*5}" for i in range(12)])

            print(f"Processed all batches. Total predictions: {len(all_preds)}")
            all_preds = np.array(all_preds)
            create_submission(all_preds, all_row_ids)  # You must implement this function!
            print(f"\nSubmission CSV saved with threshold={THRESHOLD}, TTA={tta_rounds}, batch_size={batch_size}")

        else:
            # KAGGLE SUBMISSION PIPELINE: generate submission.csv from sample_submission.csv only
            print("No test .ogg files found! Generating predictions from sample_submission.csv only.")
            sample_sub_path = "/kaggle/input/birdclef-2025/sample_submission.csv"
            sample_submission = pd.read_csv(sample_sub_path)
            row_ids = sample_submission['row_id'].values
            num_rows, num_classes = sample_submission.shape[0], sample_submission.shape[1] - 1

            # DUMMY: fill with zeros (replace with your logic if you have precomputed features/metadata)
            preds = np.zeros((num_rows, num_classes))

            # If features_dir is provided, use precomputed features for prediction
            if features_dir is not None:
                features_path = Path(features_dir)
                for i, row in sample_submission.iterrows():
                    feature_path = features_path / f"{row['row_id']}.npy"
                    if feature_path.exists():
                        try:
                            spec = np.load(feature_path)
                            if spec.shape != input_shape:
                                print(f"[✗] Invalid shape for {row['row_id']}: {spec.shape}")
                                continue
                            tta_preds = np.zeros(num_classes)
                            for t in range(tta_rounds):
                                aug_spec = np.clip(spec + np.random.normal(0, 0.01, spec.shape), 0, 1)
                                tta_preds += model.predict(aug_spec[None, ...], verbose=0)[0]
                            preds[i, :] = tta_preds / tta_rounds
                        except Exception as e:
                            print(f"[✗] Feature error for {row['row_id']}: {e}")
                            preds[i, :] = 0

            result = pd.DataFrame(preds, columns=sample_submission.columns[1:])
            result.insert(0, 'row_id', row_ids)
            result.to_csv('submission.csv', index=False)
            print("Submission file generated successfully!")

    except Exception as e:
        # Always create a fallback submission file on error
        print(f"Exception occurred: {e}\nGenerating fallback submission.csv with zeros.")
        sample_sub_path = "/kaggle/input/birdclef-2025/sample_submission.csv"
        sample_submission = pd.read_csv(sample_sub_path)
        row_ids = sample_submission['row_id'].values
        num_rows, num_classes = sample_submission.shape[0], sample_submission.shape[1] - 1
        preds = np.zeros((num_rows, num_classes))
        result = pd.DataFrame(preds, columns=sample_submission.columns[1:])
        result.insert(0, 'row_id', row_ids)
        result.to_csv('submission.csv', index=False)
        print("Fallback submission file generated.")


# Run for submission
#submission_pipeline_tta("efficientnetb2_finetuned", tta_rounds=3)
#submission_pipeline_tta("efficientnetb3_finetuned", tta_rounds=3)
submission_pipeline_tta(
    model_name="resnet50_finetuned",
    model_dir=Path("/kaggle/input/resnet50finetuned"),
    tta_rounds=3,
    batch_size=16,
    features_dir="/kaggle/input/my-precomputed-test-features",  
    input_shape=INPUT_SHAPE
)

