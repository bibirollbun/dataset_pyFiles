import pandas as pd
import numpy as np

import os


labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))


file_paths = []
base_path = '/kaggle/input/birdclef-2025/train_audio'
sub_dir = []


for dir in os.listdir(base_path):
    subdir_path = os.path.join(base_path, dir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            file_paths.append(f"{dir}/{file}")
            sub_dir.append(dir)
df = pd.DataFrame({'classes': sub_dir, 'filename': file_paths})
df.head()


from tqdm import tqdm
import librosa
from joblib import Parallel, delayed


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return np.clip(audio + noise_factor * noise, -1.0, 1.0)

def shift_pitch_audio(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def change_volume(audio, gain=1.1):
    return np.clip(audio * gain, -1.0, 1.0)



import numpy as np
from tqdm import tqdm
import os
import librosa
from joblib import Parallel, delayed

def features_extractor(file, label):
    try:
        audio, sample_rate = librosa.load(file, sr=32000)
        mfcc_list = []
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        mfcc_list.append(mfccs_scaled)
        label_list = [label]

        if np.random.rand() < 0.3:
            aug_type = np.random.choice(["noise", "volume", "pitch"], p=[0.6, 0.2, 0.2])
            if aug_type == "noise":
                augmented = add_noise(audio)
            elif aug_type == "volume":
                gain = np.random.uniform(0.8, 1.2)
                augmented = change_volume(audio, gain)
            elif aug_type == "pitch":
                steps = np.random.randint(-2, 3)
                augmented = shift_pitch_audio(audio, sr=sample_rate, n_steps=steps)
            else:
                augmented = audio

            mfccs_aug = librosa.feature.mfcc(y=augmented, sr=sample_rate, n_mfcc=40)
            mfccs_aug_scaled = (mfccs_aug - np.mean(mfccs_aug, axis=1, keepdims=True)) / (np.std(mfccs_aug, axis=1, keepdims=True) + 1e-8)
            mfcc_list.append(mfccs_aug_scaled)
            label_list.append(label)

        return mfcc_list, label_list
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return [], []



# Function to process a single file
def process_file(filename, base_path, df):
    file_path = os.path.join(base_path, filename)
    try:
        label = df.loc[df['filename'] == filename, 'classes'].values[0]
        features, labels = features_extractor(file_path, label)
        return features, labels
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return [], []



def extract_features_parallel(df_train, base_path, n_jobs=4):
    results = Parallel(n_jobs=8)(
        delayed(process_file)(filename, base_path, df_train) for filename in tqdm(df_train['filename'])
    )

    extracted_features = []
    labels = []
    for feats, lbls in results:
        extracted_features.extend(feats)
        labels.extend(lbls)

    return extracted_features, labels


base_path = '/kaggle/input/birdclef-2025/train_audio'
extracted_features, labels = extract_features_parallel(df, base_path, n_jobs=16)


import numpy as np

def pad_or_truncate(features, target_length=250): 
    if features.shape[1] > target_length:
        return features[:, :target_length] 
    else:
        padding = np.zeros((features.shape[0], target_length - features.shape[1]))
        return np.concatenate((features, padding), axis=1)

padded_features = [pad_or_truncate(f) for f in extracted_features]
padded_features = [f for f in padded_features if f is not None]

data_x = np.array(padded_features)
data_y = np.array(labels[:len(data_x)])  

print("x.shape:", data_x.shape)
print("y.shape:", data_y.shape)


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data_y)

data_y_encoded = to_categorical(integer_encoded, num_classes=206)
data_y_encoded


from sklearn.model_selection import train_test_split, StratifiedKFold


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_encoded, test_size=0.2, random_state=42)


def adjust_learning_rate(epochs):
  learning_rate = 1e-1
  if epochs > 160:
    learning_rate *= 5e-4
  elif epochs > 120:
    learning_rate *= 1e-3
  elif epochs > 80:
    learning_rate *= 5e-3
  elif epochs > 40:
    learning_rate *= 5e-2
  elif epochs >= 0:
    learning_rate *= 1e-1
  return learning_rate


import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.models import Sequential
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, TimeDistributed
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ConvLSTM2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from tensorflow.keras.models import load_model

lr_scheduler = LearningRateScheduler(adjust_learning_rate)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=5e-6)

def cnn_model(input_shape=(40, 250, 1), num_classes=206):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)

    return models.Model(inputs, outputs)



model_type = 'CNN'
save_dir = os.path.join(os.getcwd(), 'saved_models') 
model_name = f'BirdClef_{model_type}_model.{{epoch:03d}}.keras'
filepath = os.path.join(save_dir, model_name)
if not os.path.isdir(save_dir): 
    os.makedirs(save_dir) 
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, 
                              monitor='val_auc_roc', 
                              verbose=1, 
                              save_best_only=True,
                            mode='max') 
callbacks = [checkpoint, lr_reducer, lr_scheduler]
cnn = cnn_model()
cnn.compile(optimizer=Adam(learning_rate=adjust_learning_rate(0)), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='auc_roc')])

cnn.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),  shuffle=True, callbacks=callbacks)




