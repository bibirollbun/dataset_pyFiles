# !pip install --upgrade jupyter ipywidgets
# !jupyter nbextension enable --py widgetsnbextension


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def preprocess_light_curves(lc_df, meta_df, max_seq_len=162):
    """
    Preprocess PLAsTiCC light curves based on 3DSubM steps.
    Returns: Dictionary {object_id: sequence}, padded to same length.
    """

    # 1. Add Gaussian noise to flux
    lc_df['flux'] += np.random.normal(
        loc=0, scale=(lc_df['flux_err'] * 2 / 3)
    )

    # 2. Compute flux scaler
    flux_max = lc_df['flux'].max()
    flux_min = lc_df['flux'].min()
    flux_scaler = 1 / np.log2(flux_max - flux_min + 1)  # add 1 to avoid log(0)

    # 3. Normalize flux
    lc_df['flux'] *= flux_scaler

    # 4. Replace MJD with delta time (per object)
    lc_df['mjd'] = lc_df.groupby('object_id')['mjd'].transform(lambda x: x - x.min())

    # 5. Group observations by object and aggregate per-night flux
    grouped_data = {}
    for obj_id, group in tqdm(lc_df.groupby('object_id'), desc="Processing objects"):
        # Create 2D sequence: [time_steps x 6 passbands]
        sequence = []
        current_night = -10  # dummy value to group by nights (>=8 hr apart)
        night_buffer = {}
        
        for _, row in group.sort_values('mjd').iterrows():
            if row['mjd'] - current_night > 0.33:  # new night if >8 hours
                if night_buffer:
                    sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])
                night_buffer = {}
                current_night = row['mjd']
            night_buffer[row['passband']] = row['flux']

        if night_buffer:
            sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])

        # Interpolate missing passbands in each time step
        sequence = pd.DataFrame(sequence).interpolate(axis=1, limit_direction='both')

        # Replace NaNs with 0
        sequence = sequence.fillna(0).values.tolist()

        # Pad or trim to fixed length
        if len(sequence) < max_seq_len:
            pad = [[0]*6] * (max_seq_len - len(sequence))
            sequence.extend(pad)
        else:
            sequence = sequence[:max_seq_len]

        grouped_data[obj_id] = np.array(sequence)

    return grouped_data  # <-- Ensure this is typed with a normal space


train_lc = pd.read_csv("/kaggle/input/PLAsTiCC-2018/training_set.csv")
train_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/training_set_metadata.csv")

# Only use 85% for Phase 1
from sklearn.model_selection import train_test_split
object_ids = train_meta['object_id'].unique()
train_ids, val_ids = train_test_split(object_ids, test_size=0.15, random_state=42)

# Filter light curve data
train_lc_split = train_lc[train_lc['object_id'].isin(train_ids)]
val_lc_split = train_lc[train_lc['object_id'].isin(val_ids)]

# Run preprocessing
X_train_seq = preprocess_light_curves(train_lc_split, train_meta)
X_val_seq = preprocess_light_curves(val_lc_split, train_meta)  # Fixed space


from tensorflow.keras.utils import to_categorical

# Merge metadata to get class labels
train_meta = pd.read_csv("/kaggle/input/PLAsTiCC-2018/training_set_metadata.csv")
train_meta = train_meta.set_index('object_id')

# Filter labels only for the IDs present in our split
y_train = [train_meta.loc[obj_id]['target'] for obj_id in X_train_seq.keys()]
y_val = [train_meta.loc[obj_id]['target'] for obj_id in X_val_seq.keys()]

# Convert to arrays
X_train = np.array(list(X_train_seq.values()))
X_val = np.array(list(X_val_seq.values()))

# Map target classes to 0–14 range
unique_classes = sorted(train_meta['target'].unique())
class_to_index = {c: i for i, c in enumerate(unique_classes)}
y_train = np.array([class_to_index[y] for y in y_train])
y_val = np.array([class_to_index[y] for y in y_val])

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=15)
y_val = to_categorical(y_val, num_classes=15)  # <-- Fixed! (normal space)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dropout, Dense, GlobalMaxPooling1D

model = Sequential([
    Bidirectional(GRU(256, return_sequences=True), input_shape=(162, 6)),
    Dropout(0.1),
    Bidirectional(GRU(64, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='tanh'),
    Dropout(0.1),
    Dense(15, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping

# Optional: stop early if validation accuracy stops improving
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=3, 
    restore_best_weights=True
)

# Train the GRU model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    # callbacks=[early_stop],
    verbose=1
)


import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict probabilities on validation set
val_probs = model.predict(X_val)

# Convert to class predictions
val_preds = np.argmax(val_probs, axis=1)
val_true = np.argmax(y_val, axis=1)

# Map class indices back to original class labels
index_to_class = {v: k for k, v in class_to_index.items()}
val_preds_labels = [index_to_class[i] for i in val_preds]
val_true_labels = [index_to_class[i] for i in val_true]

# Classification report
print("Classification Report:")
print(classification_report(val_true_labels, val_preds_labels))

# Confusion matrix
cm = confusion_matrix(val_true_labels, val_preds_labels, labels=sorted(index_to_class.values()))

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(index_to_class.values()), yticklabels=sorted(index_to_class.values()))
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()


model.save("gru_model_phase1.h5")


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
# import tensorflow as tf
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("gru_model_phase1.h5")


# Map model class indices to actual class labels from training
index_to_class = {
    0: 6, 1: 15, 2: 16, 3: 42, 4: 52, 5: 53, 6: 62, 7: 64, 8: 65,
    9: 67, 10: 88, 11: 90, 12: 92, 13: 95, 14: 99
}


def preprocess_light_curves_test(lc_df, max_seq_len=162):
    lc_df['flux'] += np.random.normal(loc=0, scale=(lc_df['flux_err'] * 2 / 3))
    flux_max = lc_df['flux'].max()
    flux_min = lc_df['flux'].min()
    flux_scaler = 1 / np.log2(flux_max - flux_min + 1)
    lc_df['flux'] *= flux_scaler
    lc_df['mjd'] = lc_df.groupby('object_id')['mjd'].transform(lambda x: x - x.min())

    grouped_data = {}
    for obj_id, group in lc_df.groupby('object_id'):
        sequence = []
        current_night = -10
        night_buffer = {}
        for _, row in group.sort_values('mjd').iterrows():
            if row['mjd'] - current_night > 0.33:
                if night_buffer:
                    sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])
                night_buffer = {}
                current_night = row['mjd']
            night_buffer[row['passband']] = row['flux']

        if night_buffer:
            sequence.append([night_buffer.get(pb, np.nan) for pb in range(6)])

        sequence = pd.DataFrame(sequence).interpolate(axis=1, limit_direction='both')
        sequence = sequence.fillna(0).values.tolist()

        if len(sequence) < max_seq_len:
            pad = [[0]*6] * (max_seq_len - len(sequence))
            sequence.extend(pad)
        else:
            sequence = sequence[:max_seq_len]

        grouped_data[obj_id] = np.array(sequence)
    return grouped_data


import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd

# Assuming your model and preprocess function are already defined
# model = ... (load your trained model)
# def preprocess_light_curves_test(df): ...

# Directory containing test batches
batch_dir = "/kaggle/input/PLAsTiCC-2018/"
object_preds = {}
batch_files = [f"test_set_batch{i}.csv" for i in range(1, 6)]

for batch_file in tqdm(batch_files, desc="Processing batches"):
    path = os.path.join(batch_dir, batch_file)
    batch_df = pd.read_csv(path)

    test_seq_dict = preprocess_light_curves_test(batch_df)
    if not test_seq_dict:
        continue

    X_test = np.array(list(test_seq_dict.values()))
    preds = model.predict(X_test, batch_size=256)  # Optimized for T4

    for i, oid in enumerate(test_seq_dict.keys()):
        object_preds[oid] = preds[i]


