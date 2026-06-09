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


df['classes'].astype("string")


from tqdm import tqdm
import librosa
from joblib import Parallel, delayed



import numpy as np
from tqdm import tqdm
import os
import librosa
from joblib import Parallel, delayed

def features_extractor(file):
    try:
        audio, sample_rate = librosa.load(file, sr=32000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs_scaled = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None



# Function to process a single file
def process_file(filename, base_path):
    file_path = os.path.join(base_path, filename)
    features = features_extractor(file_path)
    return features


def extract_features_parallel(df_train, base_path, n_jobs=4):

    extracted_features = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(filename, base_path) for filename in tqdm(df['filename'])
    )
    extracted_features = [features for features in extracted_features if features is not None]
    
    return extracted_features

base_path = '/kaggle/input/birdclef-2025/train_audio'
extracted_features = extract_features_parallel(df, base_path, n_jobs=4)


import pickle

with open('/kaggle/working/features.pkl', 'wb') as f:
    pickle.dump(extracted_features, f)


import pickle
with open('/kaggle/input/mfcc-features/features.pkl', 'rb') as f:
    extracted_features = pickle.load(f)


import numpy as np

def pad_or_truncate(features, target_length=250): 
    if features.shape[1] > target_length:
        return features[:, :target_length] 
    else:
        padding = np.zeros((features.shape[0], target_length - features.shape[1]))
        return np.concatenate((features, padding), axis=1)

extracted_features_padded = [pad_or_truncate(features) for features in extracted_features]

extracted_features_array = np.array(extracted_features_padded)

print(f"{extracted_features_array.shape}")



data_x = extracted_features_array
data_y = df['classes']


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data_y)

data_y_encoded = to_categorical(integer_encoded, num_classes=206)
data_y_encoded


from sklearn.model_selection import train_test_split, StratifiedKFold


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y_encoded, test_size=0.3, random_state=42)


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

def cnn_model(input_shape=(20, 250, 1), num_classes=206):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

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



import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=1))
    return loss



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
cnn.compile(
        optimizer=Adam(learning_rate=adjust_learning_rate(0)), 
        loss=focal_loss(gamma=2.0, alpha=0.25), 
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='auc_roc')]
)
cnn.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),  shuffle=True, callbacks=callbacks)


import cv2
def preprocess_mfcc_to_rgb(mfcc_sample):
    resized = cv2.resize(mfcc_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    rgb = np.stack([resized] * 3, axis=-1)  
    
    rgb = rgb.astype(np.float32) / 255.0
    return rgb


def create_dataset(X, y, batch_size=32, shuffle=True):
    def _map_fn(mfcc, label):
        img = tf.numpy_function(preprocess_mfcc_to_rgb, [mfcc], tf.float32)
        img.set_shape((224, 224, 3))
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
train = create_dataset(X_train, y_train)
test = create_dataset(X_test, y_test)


from tensorflow.keras.applications import EfficientNetB0
def EfficientNet(num_classes=206):
    base_model = EfficientNetB0(include_top=False, weights='/kaggle/input/efficientb0/keras/default/1/efficientnetb0_notop.h5', input_shape=(224, 224, 3))

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


model_type = 'EfficientNetB0'
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

eff = EfficientNet()
eff.compile(
        optimizer=Adam(learning_rate=adjust_learning_rate(0)), 
        loss=focal_loss(gamma=2.0, alpha=0.25), 
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='auc_roc')]
)
eff.fit(train, epochs=200, batch_size=32, validation_data=test,  shuffle=True, callbacks=callbacks)




