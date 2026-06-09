import tensorflow_probability as tfp
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras import layers
from tf_keras import Model
from tensorflow import keras
import os
import pandas as pd

tfd = tfp.distributions
tfpl = tfp.layers


class CFG:
    n_mels = 256
    sr = 32000
    lr = 1e-3
    duration = 5  # seconds
    samples = sr * duration
    dir_path = '/kaggle/input/birdclefprocessed/train_audio'
    label_path = '/kaggle/input/birdclef-2025/train.csv'
    tax_path = '/kaggle/input/birdclef-2025/taxonomy.csv'
    train_path = '/kaggle/input/birdclef-2025/train_audio'
    sample_path = '/kaggle/input/birdclef-2025/sample_submission.csv'
    test_path = '/kaggle/input/birdclef-2025/test_soundscapes'


def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=CFG.sr)
    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CFG.n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec.T


def load_and_preprocess_audio(filename):
    # Load audio file
    filepath = os.path.join(CFG.dir_path, filename)
    audio, _ = librosa.load(filepath, sr=CFG.sr, duration=CFG.duration)
    
    # Pad or truncate to fixed length
    if len(audio) < CFG.samples:
        audio = np.pad(audio, (0, CFG.samples - len(audio)))
    else:
        audio = audio[:CFG.samples]
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=CFG.sr, n_mels=CFG.n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize to 0-1
    log_mel_spec -= log_mel_spec.min()
    log_mel_spec /= log_mel_spec.max()
    
    return log_mel_spec.T.astype(np.float32)  # shape: (time, n_mels)


df = pd.read_csv(CFG.label_path)


df.head(2)


def gen():
    for _, row in df.iterrows():
        mel_spec = load_and_preprocess_audio(row['filename'])
        label = row['label_idx']
        yield mel_spec, label

output_signature = (
    tf.TensorSpec(shape=(None, CFG.n_mels), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

# Pad to fixed length (time dimension)
MAX_TIME = 313  # Depends on mel spec parameters, ~5 seconds
def pad_to_max_time(mel_spec, label):
    mel_spec = mel_spec[:MAX_TIME, :]
    mel_spec = tf.pad(mel_spec, [[0, MAX_TIME - tf.shape(mel_spec)[0]], [0,0]])
    return mel_spec, label

dataset = dataset.map(pad_to_max_time)
dataset = dataset.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def build_model():
    inputs = layers.Input(shape=(None, CFG.n_mels))
    x1 = tfp.layers.DenseFlipout(128, activation='relu')(inputs)
    x2 = tfp.layers.DenseFlipout(64, activation='relu')(x1)
    x3 = tfp.layers.DenseFlipout(206)(x2) #206 sound classes
    outputs = tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        tfd.Bernoulli(logits=t), reinterpreted_batch_ndims=1))(x3)
    model = Model(inputs=inputs, outputs=outputs)
    return model


#model = build_model()


file_path = '/kaggle/input/birdclefprocessed/train_audio/CSA03598.wav'


# x = load_audio(file_path)


o = load_and_preprocess_audio(file_path)


o.shape




