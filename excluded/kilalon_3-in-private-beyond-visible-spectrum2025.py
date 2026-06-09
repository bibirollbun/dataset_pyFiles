import kagglehub
#kagglehub.login()
beyond_visible_spectrum_ai_for_agriculture_2025_path = kagglehub.competition_download('beyond-visible-spectrum-ai-for-agriculture-2025')
print('Data source import complete.')


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2

# Was hoping to use my relative's pc with GPU but the chance didnt come, i just left this as an if/else statement in case i ever came across it

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPU detected. Using CPU.")

npy_folder = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"
csv_path = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv"
target_shape = (32, 32, 32)
batch_size = 1
epochs = 15
model_path = "mini_model_cpu3.keras"

class DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, target_shape, npy_folder, augment=False):
        self.raw_file_paths = file_paths
        self.raw_labels = labels
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.npy_folder = npy_folder
        self.augment = augment
        self.file_paths, self.labels = self._filter_valid_files() # Filter (modify later)
        self.indexes = np.arange(len(self.file_paths))

    def _filter_valid_files(self):
        valid_paths = []
        valid_labels = []
        for path, label in tqdm(zip(self.raw_file_paths, self.raw_labels), desc="Filtering corrupt files"):
            try:
                full_path = os.path.join(self.npy_folder, path.strip())
                data = np.load(full_path)
                if data.size == 0 or len(data.shape) != 3:
                    raise ValueError("Empty or malformed data")
                valid_paths.append(path)
                valid_labels.append(label)
            except Exception as e:
                print(f"Skipping file: {path} | Reason: {e}")
        return valid_paths, valid_labels

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        X_batch = np.zeros((self.batch_size, *self.target_shape, 1), dtype=np.float32)
        for i, file_path in enumerate(batch_paths):
            try:
                full_path = os.path.join(self.npy_folder, file_path.strip())
                data = np.load(full_path)
                data = np.expand_dims(data, axis=-1)
                resized = np.zeros((*self.target_shape, 1), dtype=np.float32)
                for z in range(min(data.shape[2], self.target_shape[2])):
                    resized[:, :, z, 0] = cv2.resize(data[:, :, z], self.target_shape[:2])
                X_batch[i] = resized
            except Exception as e:
                print(f"Error batch loading: {file_path} | Reason: {e}")
                continue 

        return X_batch, np.array(batch_labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
        
def build_mini_model(input_shape=(32, 32, 32, 1)): # CNN of shape (32, 32, 32, 1) # Attempt 27 (afterupdate)
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='linear')(x)

    model = Model(inputs, output)
    return model

df = pd.read_csv(csv_path)
file_paths = df['id'].values
labels = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

train_gen = DataGenerator(X_train, y_train, batch_size, target_shape, npy_folder)
val_gen = DataGenerator(X_val, y_val, batch_size, target_shape, npy_folder)

model = build_mini_model(input_shape=(*target_shape, 1))
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
]

model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)

# Hey There! I tried a lot of modifications to this like batch size of 2, 4 and 8, i modified some stuff but the current state has the best ones, batch size 1, it is also very very cpu friendly, this model is my best of all!!!

#There is another python program I use for prediction, i will send it as well!!


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model  # type: ignore


npy_folder = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"
csv_path = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv"
model_path = "mini_model_cpu3.keras"
target_shape = (128, 128, 125)
model_input_shape = (32, 32, 32)

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"file not found: {model_path}")
model = load_model(model_path)
print("Model loaded")

def preprocess_npy(file_path, target_shape=(128, 128, 125), model_input_shape=(32, 32, 32)):
    try:
        data = np.load(file_path)

        pad_width = [(0, max(0, target_shape[i] - data.shape[i])) for i in range(3)]
        data = np.pad(data, pad_width, mode='constant')

        slices = tuple(slice(0, target_shape[i]) for i in range(3))
        data = data[slices]

        data = data[:model_input_shape[0], :model_input_shape[1], :model_input_shape[2]]

        data = data.astype(np.float32)
        max_val = np.max(data)
        if max_val > 0:
            data /= max_val

        data = data[np.newaxis, ..., np.newaxis]
        return data

    except Exception as e:
        print(f"Failed or process {file_path}: {e}")
        return None
    
def safe_predict(file_path, target_shape, model_input_shape, model):
    try:
        volume = preprocess_npy(file_path, target_shape, model_input_shape)
        if volume is None:
            return None
        return model.predict(volume, verbose=0)[0][0]
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None
    
df_test = pd.read_csv(csv_path)
df_test['id'] = df_test['id'].astype(str).str.replace('.npy', '', regex=False).str.strip()

predictions = []
skipped_indices = []

for idx, file_id in enumerate(tqdm(df_test['id'], desc="Processing test samples")):
    file_path = os.path.join(npy_folder, f"{file_id}.npy")
    
    pred = safe_predict(file_path, target_shape, model_input_shape, model)
    
    if pred is None:
        predictions.append(None)
        skipped_indices.append(idx)
        continue

    predictions.append(pred)
    
num_skipped = len(skipped_indices)
if num_skipped > 0:
    print( (num_skipped) + " samples failed to process. Filling with mean prediction.")
    valid_preds = [p for p in predictions if p is not None]
    fallback_value = np.mean(valid_preds) if valid_preds else 0.0
    for i in skipped_indices:
        predictions[i] = fallback_value
        
df_test['label'] = predictions
df_test.to_csv("predictions.csv", index=False)
print("Predictions saved Finnaly!")


