import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def audio_to_melspectrogram_image(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000, duration=5, img_size=256):
    audio_path = audio_path.numpy().decode("utf-8")
    # Load the first 'duration' seconds of audio
    y, sr = librosa.load(audio_path, sr = sr)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, 
    n_mels=128, fmax=sr // 2)
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = 255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_norm = mel_spec_norm.astype(np.float32)
    mel_image = Image.fromarray(mel_spec_norm)
    mel_image = mel_image.resize((img_size, img_size), Image.LANCZOS)
    
    # Convert to 3-channel image
    mel_image = np.stack([mel_image] * 3, axis=-1)
    return mel_image

def preprocess(file_path):
    features = tf.py_function(
            func=audio_to_melspectrogram_image, 
            inp=[file_path], 
            Tout=tf.float32
        )
    
    return features



import os
import pandas as pd
df =pd.read_csv('/kaggle/input/audio-files/audio.csv')
bird_classes = os.listdir('/kaggle/input/birdclef-2024/train_audio')
labels = []
files = []
for i in df['audio']:
    files.append(i)
    bc = i.split('/')[-2]
    labels.append(bird_classes.index(bc))


import random
random.seed(123)
random.shuffle(files)
random.seed(123)
random.shuffle(labels)
train_sample = int(len(files)*0.9)
training_files = files[:train_sample]
training_labels = labels[:train_sample]

testing_files = files[train_sample:]
testing_labels = labels[train_sample:]


import tensorflow as tf
training_dataset_files = tf.data.Dataset.from_tensor_slices(training_files)
training_dataset_files = training_dataset_files.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
training_dataset_labels = tf.data.Dataset.from_tensor_slices(training_labels)
training_data = tf.data.Dataset.zip((training_dataset_files, training_dataset_labels))
training_data = training_data.map(lambda audio, label: (tf.ensure_shape(audio, (256,256,3)), 
                                          tf.ensure_shape(label, ())))
                                          
training_data = training_data.batch(64)

testing_dataset_files = tf.data.Dataset.from_tensor_slices(testing_files)
testing_dataset_files = testing_dataset_files.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
testing_dataset_labels = tf.data.Dataset.from_tensor_slices(testing_labels)
testing_data = tf.data.Dataset.zip((testing_dataset_files, testing_dataset_labels))
testing_data = testing_data.map(lambda audio, label: (tf.ensure_shape(audio, (256,256,3)), 
                                          tf.ensure_shape(label, ())))
                                          
testing_data = testing_data.batch(64)





import tensorflow as tf
import numpy as np

class CosineAnnealingWithWarmup(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, warmup_epochs=5, peak_lr=1e-4):
        super().__init__()
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.peak_lr = peak_lr
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # Keras epoch starts from 0
        new_lr = self.compute_lr()
        self.model.optimizer.learning_rate.assign(new_lr)
        print(f"Epoch {self.current_epoch}: Learning Rate = {new_lr:.6f}")

    def compute_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # Linear Warmup: Increase LR linearly to peak_lr
            return (self.peak_lr / self.warmup_epochs) * self.current_epoch
        else:
            # Cosine Annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return 0.5 * self.peak_lr * (1 + np.cos(np.pi * progress))


from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint_epoch_{epoch:02d}.keras",  # Save model after each epoch
    save_weights_only=False,  # Set to True if you only want to save weights
    save_best_only=False,  # Set to True to save only the best model based on validation loss
    verbose=1
)


import tensorflow as tf
import keras_cv 
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input


# Ensure bird_classes is defined
num_classes = len(bird_classes)  

# Load base model
base_model = EfficientNetV2B0(include_top=False)
base_model.trainable = False  # Freeze for feature extraction

# Define Model
input_shape = (256, 256, 3)
inputs = layers.Input(shape=input_shape, name="input_layer")
x = tf.keras.layers.RandomFlip(mode="horizontal")(inputs)  # Horizontal Flip
x = keras_cv.layers.RandomCutout(height_factor=0.2, width_factor=0.2)(x)
x = base_model(x, training=False)  # Keep batchnorm frozen
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)
model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

total_epochs = 16
warmup_epochs = 5
peak_lr = 1e-4
cosine_warmup_callback = CosineAnnealingWithWarmup(total_epochs, warmup_epochs, peak_lr)

# Early stopping callback (Accuracy monitoring)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=7, mode="max", restore_best_weights=True
)

'''
model.fit(
    training_data,
    validation_data=testing_data,
    epochs=total_epochs,
    batch_size=64,
    callbacks=[cosine_warmup_callback, early_stopping,checkpoint_callback]
)
'''


'''
base_model.trainable = True
for layer in model_2_base_model.layers[:-10]:
  layer.trainable = False
'''


import tensorflow as tf
model = tf.keras.models.load_model(
    '/kaggle/input/best-model/model_checkpoint_epochft_02.keras'
)


from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint_epochft_{epoch:02d}.keras",  # Save model after each epoch
    save_weights_only=False,  # Set to True if you only want to save weights
    save_best_only=False,  # Set to True to save only the best model based on validation loss
    verbose=1
)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # lr is 10x lower than before for fine-tuning
                metrics=["accuracy"])


'''
model.fit(
    training_data,
    validation_data=testing_data,
    epochs=5,
    batch_size=64,
    callbacks=[checkpoint_callback]
)
'''


fin_model = tf.keras.models.load_model(
    '/kaggle/input/best-weight-36/model_checkpoint_epochft_05.keras'
)
for i,o in testing_data.take(1):
    for audio,label in zip(i,o):
        res = model.predict(np.expand_dims(audio, axis=0))
        p_index = np.argmax(res)
        print('predicted result: ', bird_classes[p_index])
        print('actual result: ', bird_classes[label])

