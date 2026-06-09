import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


AUDIO_DIR = "/kaggle/input/freesound-audio-tagging/audio_train"
CSV_PATH = "/kaggle/input/freesound-audio-tagging/train.csv"     
SAMPLE_RATE = 22050
N_MELs = 90


def load_dataset_from_csv(audio_dir, csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} entries")
    print(f"CSV columns: {df.columns.tolist()}")
    
    filename_col = 'fname'
    label_col = 'label'
    
    classes = sorted(df[label_col].unique())
    print(f"Found {len(classes)} classes: {classes}")
    
    features = []
    labels = []
    file_paths = []
    durations = []
    missing_files = []
    
    # Create label to index mapping
    class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}
    print("Class to index mapping:", class_to_index)
    
    for _, row in df.iterrows():
        filename = row[filename_col]
        class_name = row[label_col]
        file_path = os.path.join(audio_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            missing_files.append(filename)
            continue
            
        # Get duration for analysis
        try:
            duration = librosa.get_duration(filename=file_path)
            durations.append(duration)
            
            # Process audio file
            audio, sr = load_and_preprocess_audio(file_path)
            if audio is None:
                continue
                
            feature = extract_features_from_audio(audio, sr, N_MELs)
            
            if feature is not None:
                features.append(feature)
                labels.append(class_to_index[class_name])
                file_paths.append(file_path)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files from CSV not found in audio folder")
        print("Sample missing files:", missing_files[:5])
    
    print(f"\nSuccessfully processed {len(features)} files")
    print(f"Audio Duration Statistics:")
    print(f"Min: {min(durations):.2f}s, Max: {max(durations):.2f}s, Mean: {np.mean(durations):.2f}s")
    
    return np.array(features), np.array(labels), file_paths, classes, durations


def load_and_preprocess_audio(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def extract_features_from_audio(audio, sr, n_mels=64):
    
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=4096, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spectrogram)
    log_mel_spec = normalize(log_mel_spec)
    
    if log_mel_spec.shape[1] < 300:
        pad_width = 300 - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif log_mel_spec.shape[1] > 300:
        log_mel_spec = log_mel_spec[:, :300]  

    return log_mel_spec


def normalize(spec):
    std = np.std(spec)
    if std == 0:
        std = 1e-10
    return (spec - np.mean(spec)) / np.std(spec)


def create_variable_length_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # First conv block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def augment_features(features):
    if np.random.random() > 0.5:
        max_mask_pct = 0.2
        mask_size = int(features.shape[1] * max_mask_pct * np.random.random())
        mask_start = np.random.randint(0, features.shape[1] - mask_size)
        features[:, mask_start:mask_start + mask_size] = 0
    
    # Frequency masking
    if np.random.random() > 0.5:
        max_mask_pct = 0.15
        mask_size = int(features.shape[0] * max_mask_pct * np.random.random())
        mask_start = np.random.randint(0, features.shape[0] - mask_size)
        features[mask_start:mask_start + mask_size, :] = 0
    
    return features


def create_data_generator(features, labels, batch_size=32, augment=False):
    num_samples = len(features)
    
    while True:
        indices = np.random.permutation(num_samples)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_features = []
            batch_labels = []
            
            for idx in batch_indices:
                feature = features[idx].copy()

                if augment:
                    feature = augment_features(feature)
                
                batch_features.append(feature)
                batch_labels.append(labels[idx])
            
            # Convert to numpy arrays and add channel dimension
            batch_features = np.array(batch_features)
            batch_features = batch_features[..., np.newaxis]  # Add channel dimension
            
            batch_labels = keras.utils.to_categorical(batch_labels, num_classes=len(np.unique(labels)))
            
            yield batch_features, batch_labels


X, y, file_paths, classes, durations = load_dataset_from_csv(AUDIO_DIR, CSV_PATH)


class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)


y_categorical = keras.utils.to_categorical(y, num_classes=len(classes))


X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y_categorical, file_paths, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")


input_shape = (X[0].shape[0], X[0].shape[1], 1)
print(f"Input shape: {input_shape}")


model = create_variable_length_model(input_shape, len(classes))
    
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy',
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        patience=8,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_audio_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
]

model.summary()


batch_size = 32
train_gen = create_data_generator(X_train, np.argmax(y_train, axis=1), 
                                batch_size=batch_size, augment=True)
val_gen = create_data_generator(X_val, np.argmax(y_val, axis=1), 
                              batch_size=batch_size)

# Calculate steps per epoch
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size


print("Training model...")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1 )


print("\nEvaluating on test set...")
X_test_processed = np.array([x[..., np.newaxis] for x in X_test])

y_pred = model.predict(X_test_processed)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
print(f"FINAL TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


TEST_AUDIO_DIR = "/kaggle/input/freesound-audio-tagging/audio_test"
TEST_CSV_PATH = "/kaggle/input/freesound-audio-tagging/sample_submission.csv"


X_Test, _, file_paths, _, durations = load_dataset_from_csv(TEST_AUDIO_DIR, TEST_CSV_PATH)


input_shape = (X_Test[0].shape[0], X_Test[0].shape[1], 1)
print(f"Input shape: {input_shape}")


fnames = [os.path.basename(p) for p in file_paths]


def predict_and_save(model, features, fnames, classes, output_csv="submission.csv"):
    # Predict
    preds = model.predict(features)
    predicted_indices = np.argmax(preds, axis=1)
    predicted_labels = [classes[i] for i in predicted_indices]

    # Save to CSV
    submission_df = pd.DataFrame({
        'fname': fnames,
        'label': predicted_labels
    })
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


Test_features = np.array(X_Test)
Test_features = Test_features[..., np.newaxis]


predict_and_save(model, Test_features, fnames, classes)




