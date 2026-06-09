import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json

from pathlib import Path

import gc

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Input, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization, Reshape, GRU
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset_path = Path("../input/birdclef-2021/")

MAX_LENGTH = 40

train_metadata_file = dataset_path / "train_metadata.csv"
train_dir_short = dataset_path / "train_short_audio"
train_data_df = pd.read_csv(train_metadata_file)
train_data_df.head()


TEST_AUDIO_PATH = os.path.join(dataset_path, "test_soundscapes")
test_meta = pd.read_csv(os.path.join(dataset_path, "test.csv"))
use_train_as_test = len(test_meta) < 10
audio_source = os.path.join(dataset_path, "train_soundscapes") if use_train_as_test else TEST_AUDIO_PATH
if use_train_as_test:
    print("Case when train is the test")
    test_meta = pd.read_csv(os.path.join(dataset_path, "train_soundscape_labels.csv"))

print(f"Total segments to process: {len(test_meta)}")



train_data_df = (train_data_df[["primary_label"]]
                 .assign(path = train_data_df.apply(
                    lambda row: Path(dataset_path) / row["primary_label"] / row["filename"],
                    axis=1
                )))
train_data_df



LABEL_TO_NUM = {l: n for n, l in enumerate(train_data_df.primary_label.unique())}
NUM_TO_LABEL = {n: l for l, n in LABEL_TO_NUM.items()}
NUM_CLASSES = len(LABEL_TO_NUM)
NUM_CLASSES


encoder = LabelEncoder() 

labels = encoder.fit_transform(train_data_df["primary_label"].unique())
indexes = encoder.inverse_transform(labels)


value_counts = train_data_df["primary_label"].value_counts()

truncate_counts = value_counts.map(lambda x: x if x <= MAX_LENGTH else MAX_LENGTH)

value_counts


truncate_counts.sum()


from pathlib import Path

dirs = os.listdir(train_dir_short)

files_by_label = {}


for label in tqdm(dirs):
    folder = train_dir_short / label
    sorted_files = sorted(folder.iterdir(), key=lambda f: f.stat().st_size)
    files_by_label[label] = [str(f) for f in sorted_files[:MAX_LENGTH]]


SAMPLE_RATE = 32000
N_MFCC = 40
HOP_LENGTH = 512 


def load_and_convert_to_mfcc(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).astype(np.float16)
    return mfcc


from joblib import Parallel, delayed
import os

train_metadata = pd.read_csv(dataset_path / "train_metadata.csv")

train_data = []
not_in = []

def process_file(file_path, label):
    if os.path.exists(file_path):
        mfcc = load_and_convert_to_mfcc(file_path)
        return {'mfcc': mfcc, 'label': label}
    else:
        print(f"Файл не найден: {file_path}")
        return None

results = Parallel(n_jobs=-1)(
    delayed(process_file)(file_path, label)
    for label, files in tqdm(files_by_label.items())
    for file_path in files
)

# Supprimer les None (fichiers non trouvés)
train_data = [r for r in results if r is not None]
print(f"len train data: {len(train_data)}")


train_data[0]


data_lens = sorted([i['mfcc'].shape[1] for i in train_data])

plt.plot(data_lens)
plt.show()

stats = pd.Series(data_lens)

print(stats.describe())


MAX_LEN = 3000
greater_then_max_len = [i for i in data_lens if i > MAX_LEN]
amount_greater_max = len(greater_then_max_len)

print(amount_greater_max)


def process_padding_mfcc(d):
    mfcc = librosa.util.fix_length(d['mfcc'], size=MAX_LEN, axis=1)
    mfcc = mfcc[..., np.newaxis]
    return mfcc, d['label']

results = Parallel(n_jobs=-1)(
    delayed(process_padding_mfcc)(d) for d in tqdm(train_data, desc="Processing MFCCs")
)

# Séparer MFCCs et labels
mfccs_padded, labels = zip(*results)


for i, d in enumerate(train_data):
    d['mfcc'] = mfccs_padded[i]  # remplace MFCC par le pad/crop + channel


print(len(mfccs_padded[0][0]))


train_data[0]['mfcc'].shape


labels = [d['label'] for d in train_data]
labels_categorical = pd.get_dummies(labels).values

print(labels_categorical.shape)
print(labels_categorical)


labels_as_nums = [LABEL_TO_NUM[i['label']] for i in train_data]
labels_categorical = to_categorical(labels_as_nums, num_classes=NUM_CLASSES)
print(f"labels_categorical shape = {labels_categorical.shape}")
print(f"labels_categorical shape = {labels_categorical}")


X = np.stack([i['mfcc'] for i in train_data], axis=0)
X_train, X_val, y_train, y_val = train_test_split(
    X, labels_categorical, test_size=0.3, random_state=42
)


# import tensorflow_addons as tfa

# f1_score = tfa.metrics.F1Score(num_classes=NUM_CLASSES, average='macro', threshold=0.5)


kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64  
conv_depth_3 = 128
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 512


batch_size = 32
num_epochs = 3


inp = Input(shape=(X_train.shape[1:]))

# CNN feature extractor
x = Convolution2D(conv_depth_1, (kernel_size + 2, kernel_size + 2), activation='relu', padding='same')(inp)
x = BatchNormalization()(x)
x = Convolution2D(conv_depth_1, (kernel_size + 2, kernel_size + 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((pool_size,pool_size))(x)
x = Dropout(0.25)(x)    
    

x = Convolution2D(conv_depth_1, (kernel_size + 1, kernel_size + 1), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Convolution2D(conv_depth_1, (kernel_size + 1, kernel_size + 1), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((pool_size,pool_size))(x)
x = Dropout(0.25)(x)

x = Convolution2D(conv_depth_2, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Convolution2D(conv_depth_2, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((pool_size,pool_size))(x)
x = Dropout(0.3)(x)


# Reshape correctement avec int()
x = GlobalAveragePooling2D()(x)

# Recurrent part
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
# Dense output
output = Dense(NUM_CLASSES, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=output)

model.summary()

early_stop = EarlyStopping(
monitor='val_loss',
patience=5,
verbose=1)

model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']) 
history = model.fit(
X_train, y_train,
batch_size=batch_size,
epochs=num_epochs,
verbose=1,
validation_data=[X_val, y_val],
callbacks=[early_stop])


model.evaluate(X_val, y_val)


SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SEGMENT_SECONDS * SAMPLE_RATE

X_test_segments = []
row_ids = []

test_dir = Path(audio_source)

# Lister uniquement les fichiers .ogg
ogg_files = [f for f in test_dir.iterdir() if f.suffix.lower() == ".ogg"]

print(f"Nombre de fichiers .ogg à traiter : {len(ogg_files)}")

for f in tqdm(ogg_files, desc="Traitement segments"):
    audio, _ = librosa.load(f, sr=SAMPLE_RATE)
    num_segments = max(1, len(audio) // SEGMENT_SAMPLES)

    for i in range(num_segments):
        start = i * SEGMENT_SAMPLES
        end = start + SEGMENT_SAMPLES
        segment_audio = audio[start:end]

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=segment_audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
        )
        mfcc = librosa.util.fix_length(mfcc, size=MAX_LEN, axis=1)
        mfcc = mfcc[..., np.newaxis]  # ajouter dimension channel

        X_test_segments.append(mfcc)
        row_ids.append(f"{'_'.join(f.stem.split('_')[0:2])}_{i*SEGMENT_SECONDS + 5}")

# Convertir en array numpy
if len(X_test_segments) > 0:
    X_test_segments = np.stack(X_test_segments, axis=0)
else:
    raise ValueError("⚠️ Aucun segment à traiter !")

# -------------------------------
# Prédictions avec seuil
# -------------------------------
lower_limit = 0.25  # seuil pour sélectionner plusieurs oiseaux

res_birds = []

preds_segments = model.predict(X_test_segments, batch_size=32, verbose=1)

for pred in preds_segments:
    selected_indices = np.where(pred > lower_limit)[0]
    predicted_labels = [NUM_TO_LABEL[i] for i in selected_indices]
    birds = ' '.join(predicted_labels) if predicted_labels else 'nocall'
    res_birds.append(birds)

# -------------------------------
# Créer le CSV de submission
# -------------------------------
submission_df = pd.DataFrame({
    "row_id": row_ids,
    "birds": res_birds
})

submission_df.to_csv("submission.csv", index=False)
print("Fichier submission.csv généré avec seuil et row_id correct !")







