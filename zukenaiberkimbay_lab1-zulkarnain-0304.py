import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

TRAIN_CSV = '../input/freesound-audio-tagging/train.csv'
TRAIN_DIR = '../input/freesound-audio-tagging/audio_train/'
TEST_DIR = '../input/freesound-audio-tagging/audio_test/'
SAMPLE_SUB = '../input/freesound-audio-tagging/sample_submission.csv'

N_MELS = 128
TIME_STEPS = 256
SR = 32000
BATCH_SIZE = 32
EPOCHS = 35
NUM_CLASSES = 41


df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(SAMPLE_SUB)

unique_labels = sorted(df_train['label'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

df_train['label_idx'] = df_train['label'].map(label2id)

def read_audio(filename, audio_dir):
    file_path = os.path.join(audio_dir, filename)
    y, _ = librosa.load(file_path, sr=SR)
    y, _ = librosa.effects.trim(y)
    
    if len(y) < TIME_STEPS * 512:
        padding = TIME_STEPS * 512 - len(y)
        y = np.pad(y, (0, padding), 'constant')
        
    mels = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=2048, hop_length=512)
    mels = librosa.power_to_db(mels, ref=np.max)
    
    mels = (mels - mels.min()) / (mels.max() - mels.min() + 1e-6)
    return mels.astype(np.float32)

X_all = []
for fname in tqdm(df_train['fname'].values):
    X_all.append(read_audio(fname, TRAIN_DIR))

y_all = df_train['label_idx'].values

X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_data, y_data, batch_size=32, dim=(128, 256), n_classes=41, shuffle=True, augment=False):
        self.dim = dim
        self.batch_size = batch_size
        self.y_data = y_data
        self.X_data = X_data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size, self.n_classes))

        for i, idx in enumerate(indexes):
            spec = self.X_data[idx]
            label = self.y_data[idx]
            
            spec_len = spec.shape[1]
            if spec_len > self.dim[1]:
                if self.augment:
                    start = np.random.randint(0, spec_len - self.dim[1])
                else:
                    start = (spec_len - self.dim[1]) // 2
                crop = spec[:, start:start+self.dim[1]]
            else:
                padding = self.dim[1] - spec_len
                crop = np.pad(spec, ((0,0), (0, padding)), 'constant')
            
            img = np.stack([crop, crop, crop], axis=-1)
            
            X[i,] = preprocess_input(img * 255) 
            y[i,] = to_categorical(label, num_classes=self.n_classes)

        if self.augment and np.random.random() > 0.3:
            lam = np.random.beta(0.2, 0.2, self.batch_size)
            X2 = X[::-1]
            y2 = y[::-1]
            lam = lam.reshape(self.batch_size, 1, 1, 1)
            X = lam * X + (1 - lam) * X2
            lam = lam.reshape(self.batch_size, 1)
            y = lam * y + (1 - lam) * y2

        return X, y

train_gen = DataGenerator(X_train_raw, y_train, BATCH_SIZE, (N_MELS, TIME_STEPS), NUM_CLASSES, augment=True)
val_gen = DataGenerator(X_val_raw, y_val, BATCH_SIZE, (N_MELS, TIME_STEPS), NUM_CLASSES, augment=False, shuffle=False)



def build_model():
    base_model = MobileNetV2(input_shape=(N_MELS, TIME_STEPS, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    
    inp = Input(shape=(N_MELS, TIME_STEPS, 3))
    x = base_model(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)



test_files = df_test['fname'].values
X_test = []

for fname in tqdm(test_files):
    X_test.append(read_audio(fname, TEST_DIR))

test_predictions = []

for spec in tqdm(X_test):
    crops = []
    spec_len = spec.shape[1]
    
    starts = []
    if spec_len > TIME_STEPS:
        starts = [0, (spec_len - TIME_STEPS)//2, spec_len - TIME_STEPS]
    else:
        starts = [0]
        
    for start in starts:
        if spec_len > TIME_STEPS:
            c = spec[:, start:start+TIME_STEPS]
        else:
            padding = TIME_STEPS - spec_len
            c = np.pad(spec, ((0,0), (0, padding)), 'constant')
            
        img = np.stack([c, c, c], axis=-1)
        crops.append(preprocess_input(img * 255))
    
    crops = np.array(crops)
    preds = model.predict(crops, verbose=0)
    avg_pred = preds.mean(axis=0)
    test_predictions.append(avg_pred)

test_predictions = np.array(test_predictions)

top3_labels = []
for pred in test_predictions:
    top3_idx = pred.argsort()[-3:][::-1]
    top3_names = [id2label[idx] for idx in top3_idx]
    top3_labels.append(" ".join(top3_names))

df_test['label'] = top3_labels
df_test.to_csv('submission.csv', index=False)


