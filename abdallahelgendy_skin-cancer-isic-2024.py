import os, io
import numpy as np
import pandas as pd
import h5py
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupKFold

DATA_DIR = "/kaggle/input/isic-2024-challenge"
TRAIN_H5 = f"{DATA_DIR}/train-image.hdf5"
TEST_H5  = f"{DATA_DIR}/test-image.hdf5"

train_df = pd.read_csv(f"{DATA_DIR}/train-metadata.csv", low_memory=False)
test_df  = pd.read_csv(f"{DATA_DIR}/test-metadata.csv", low_memory=False)
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("train:", train_df.shape, "test:", test_df.shape, "sample:", sample_sub.shape)



gkf = GroupKFold(n_splits=5)
fold = 0

tr_idx, va_idx = list(gkf.split(train_df, train_df["target"], groups=train_df["patient_id"]))[fold]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print(tr_df.shape, va_df.shape)
print("pos rate:", tr_df["target"].mean(), va_df["target"].mean())



def read_image_from_h5(h5_file, isic_id, img_size=(224,224)):
    x = h5_file[isic_id][()]
    img_bytes = x.tobytes() if isinstance(x, np.ndarray) else bytes(x)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(img_size)
    return np.asarray(img, dtype=np.uint8)



class H5Sequence(keras.utils.Sequence):
    def __init__(self, df, h5_path, batch_size=64, img_size=(224,224), shuffle=False, is_train=True):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.indices = np.arange(len(self.df))
        self.h5 = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def _get_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_ids = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = self.df.iloc[batch_ids]

        h5f = self._get_h5()
        X = np.zeros((len(batch), self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        for i, isic_id in enumerate(batch["isic_id"].values):
            X[i] = read_image_from_h5(h5f, isic_id, self.img_size)

        # EfficientNet preprocessing (أفضل من /255)
        X = keras.applications.efficientnet.preprocess_input(X.astype(np.float32))

        if self.is_train:
            y = batch["target"].values.astype(np.float32)
            return X, y
        else:
            return X


IMG_SIZE = (224, 224)

weights_setting = None  # change to "imagenet" if Internet is ON

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights=weights_setting,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")]
)

model.summary()



batch_size = 64

train_seq = H5Sequence(tr_df, TRAIN_H5, batch_size=batch_size, img_size=IMG_SIZE, shuffle=True, is_train=True)
valid_seq = H5Sequence(va_df, TRAIN_H5, batch_size=batch_size, img_size=IMG_SIZE, shuffle=False, is_train=True)

steps_per_epoch = len(train_seq)
val_steps = len(valid_seq)

pos = tr_df["target"].sum()
neg = len(tr_df) - pos
class_weight = {0: 1.0, 1: float(neg / max(pos, 1.0))}
print("class_weight:", class_weight)

callbacks = [
    keras.callbacks.ModelCheckpoint("best.keras", monitor="val_auc", mode="max", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
]

history = model.fit(
    train_seq,
    validation_data=valid_seq,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)



test_seq = H5Sequence(test_df, TEST_H5, batch_size=128, img_size=IMG_SIZE, shuffle=False, is_train=False)

preds = model.predict(test_seq, steps=len(test_seq), verbose=1).reshape(-1)

sub = sample_sub.copy()
sub["target"] = preds
sub.to_csv("submission.csv", index=False)

print("Saved: submission.csv", preds.min(), preds.max())
sub.head()



sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved: /kaggle/working/submission.csv")



import os, glob

print("Files in /kaggle/working:")
for f in sorted(os.listdir("/kaggle/working")):
    print(" -", f)

print("\nSubmission exists?", os.path.exists("/kaggle/working/submission.csv"))
print("Size:", os.path.getsize("/kaggle/working/submission.csv") if os.path.exists("/kaggle/working/submission.csv") else None)



model.save("/kaggle/working/model_final.keras")
print("Saved: /kaggle/working/model_final.keras")



#


#




