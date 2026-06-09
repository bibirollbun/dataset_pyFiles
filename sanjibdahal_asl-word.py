# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
counter = 0
print(counter)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        counter = counter + 1
        if counter == 6:
            break
    if counter == 6:
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


/kaggle/input/asl-signs/sign_to_prediction_index_map.json
/kaggle/input/asl-signs/train.csv
/kaggle/input/asl-signs/train_landmark_files/36257/3762317508.parquet
/kaggle/input/asl-signs/train_landmark_files/36257/1613088982.parquet


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


!pip install mediapipe


# Mediapipe for real-time inference
import cv2
import mediapipe as mp
from collections import deque


DATA_DIR = '/kaggle/input/asl-signs/train_landmark_files'
CSV_PATH = '/kaggle/input/asl-signs/train.csv'
SEQUENCE_LENGTH = 30
FEATURE_DIM = (33 + 468 + 21 + 21) * 3  # x, y, z for pose, face, left & right hand
EPOCHS = 50
BATCH_SIZE = 64
TEST_SIZE = 0.1
RANDOM_STATE = 42
# /kaggle/input/asl-signs/sign_to_prediction_index_map.json
# /kaggle/input/asl-signs/train.csv
# /kaggle/input/asl-signs/train_landmark_files/36257/3762317508.parquet


#%% [markdown]
# 2. Load Metadata and Prepare Label Encoder
#%%

metadata = pd.read_csv(CSV_PATH)
le = LabelEncoder()
metadata['label'] = le.fit_transform(metadata['sign'])
num_classes = metadata['label'].nunique()
print(f"Loaded {len(metadata)} sequences across {num_classes} signs.")


#%% [markdown]
# 3. Feature Extraction Helpers
#%%

def extract_sequence_from_parquet(path, seq_len=SEQUENCE_LENGTH, feat_dim=FEATURE_DIM):
    df = pd.read_parquet(path)
    seq = []
    grouped = df.groupby('frame')
    for frame_no, group in grouped:
        arr = np.zeros((feat_dim,), dtype=np.float32)
        idx = 0
        for t, count in [('pose', 33), ('face', 468), ('left_hand', 21), ('right_hand', 21)]:
            subset = group[group['type'] == t]
            coords = subset.sort_values('landmark_index')[['x','y','z']].to_numpy()
            flat = coords.flatten() if len(coords) == count else np.zeros((count*3,),)
            arr[idx:idx+count*3] = flat
            idx += count*3
        seq.append(arr)
    if len(seq) >= seq_len:
        seq = seq[:seq_len]
    else:
        for _ in range(seq_len - len(seq)):
            seq.append(np.zeros((feat_dim,),))
    return np.stack(seq)


#%% [markdown]
# 4. Build Dataset Arrays
#%%

# X = []
# Y = []
# for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
#     p = row['participant_id']
#     s = row['sequence_id']
#     path = os.path.join(DATA_DIR, str(p), f"{s}.parquet")
#     if os.path.exists(path):
#         seq_arr = extract_sequence_from_parquet(path)
#         X.append(seq_arr)
#         Y.append(row['label'])
#     else:
#         print(f"Missing file: {path}")
# X = np.array(X)
# Y = tf.keras.utils.to_categorical(Y, num_classes=num_classes)
# print("X shape:", X.shape, "Y shape:", Y.shape)

from concurrent.futures import ThreadPoolExecutor

def process_row(row):
    p = row['participant_id']
    s = row['sequence_id']
    path = os.path.join(DATA_DIR, str(p), f"{s}.parquet")
    if os.path.exists(path):
        try:
            arr = extract_sequence_from_parquet(path)
            return arr, row['label']
        except Exception as e:
            print(f"Failed reading {path}: {e}")
            return None
    return None

def process_chunk(chunk_df):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_row, [row for _, row in chunk_df.iterrows()]), total=len(chunk_df)))
    results = [r for r in results if r is not None]
    if results:
        X_chunk, Y_chunk = zip(*results)
        return np.array(X_chunk), tf.keras.utils.to_categorical(Y_chunk, num_classes=num_classes)
    else:
        return np.empty((0, SEQ_LEN, NUM_FEATURES)), np.empty((0, num_classes))  # Adjust shape as needed

# Split metadata into N chunks
chunk_size = 500  # Try small chunks first
chunks = [metadata.iloc[i:i + chunk_size] for i in range(0, len(metadata), chunk_size)]

X_parts, Y_parts = [], []

for i, chunk in enumerate(chunks):
    print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
    X_chunk, Y_chunk = process_chunk(chunk)
    X_parts.append(X_chunk)
    Y_parts.append(Y_chunk)

# Merge all processed chunks
X = np.concatenate(X_parts, axis=0)
Y = np.concatenate(Y_parts, axis=0)

print("Final X shape:", X.shape)
print("Final Y shape:", Y.shape)


print(Y_parts)


data = np.load('cached_asl_data.npz')
X, Y = data['X'], data['Y']
# df = pd.DataFrame(np.array(X, Y))
# print(df)
labels = np.argmax(Y, axis=1)
df = pd.DataFrame({"label": labels})
print(df.head())


#%% [markdown]
# 5. Train/Test Split
#%%

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y)
print("Train:", X_train.shape, Y_train.shape)
print("Val:", X_val.shape, Y_val.shape)


#%% [markdown]
# 6. Model Definition
#%%

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


#%% [markdown]
# 7. Training
#%%

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    TensorBoard(log_dir='./logs')
]
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# Save trained model
model.save('asl_lstm_parquet.h5')


model.predict(X_val)

