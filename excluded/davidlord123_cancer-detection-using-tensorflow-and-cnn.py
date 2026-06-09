!pip install ipython-autotime
%load_ext autotime


import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import glob
import tensorflow as tf


# utils/datasets.py


def build_image_dfs(
    root_dir: str,
    train_sub: str       = "train",          # folder with labelled images
    test_sub: str        = "test",           # folder with unlabelled images
    label_csv: str       = "train_labels.csv",
    id_col: str          = "id",
    label_col: str       = "label",
    ext: str             = ".tif",
    val_split: float     = 0.2,
    stratify: bool       = True,
    seed: int            = 42,
):
    """
    Returns train_df, val_df, test_df  (all with columns: id, filepath, [label])
    """

    root = Path(root_dir)

    # ---------- labelled training images ----------
    train_dir = root / train_sub
    train_paths = list(train_dir.glob(f"*{ext}"))
    df = pd.DataFrame({"filepath": train_paths})
    df[id_col] = df["filepath"].apply(lambda p: p.stem)

    labels = pd.read_csv(root / label_csv, dtype={label_col: "uint8"})
    df = df.merge(labels[[id_col, label_col]], on=id_col, how="left")

    if df[label_col].isna().any():
        raise ValueError("Some images in the folder have no label in the CSV.")

    # ---------- optional train/val split ----------
    if val_split > 0:
        strat = df[label_col] if stratify else None
        train_df, val_df = train_test_split(
            df, test_size=val_split, stratify=strat, random_state=seed
        )
    else:
        train_df, val_df = df, pd.DataFrame(columns=df.columns)

    # ---------- unlabelled test images ----------
    test_dir = root / test_sub
    test_paths = list(test_dir.glob(f"*{ext}"))
    test_df = pd.DataFrame({"filepath": test_paths})
    test_df[id_col] = test_df["filepath"].apply(lambda p: p.stem)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df




train_df, val_df, test_df = build_image_dfs(
    root_dir="/kaggle/input/histopathologic-cancer-detection",
    val_split=False,
)

TRAIN_DIR = Path("/kaggle/input/histopathologic-cancer-detection/train")
TEST_DIR = Path("/kaggle/input/histopathologic-cancer-detection/test")
train_df['filepath'] = train_df['id'].apply(lambda x: str(TRAIN_DIR / f"{x}.tif"))
test_df['filepath'] = test_df['id'].apply(lambda x: str(TEST_DIR / f"{x}.tif"))

print(train_df.head())


print("Number of duplicate rows in training set:", train_df.duplicated().sum(),"\n")
print("Number of duplicate rows in testing set:", test_df.duplicated().sum(),"\n")


from PIL import Image
import random
from pathlib import Path

# ------------------------------------------------------------------
def show_df_samples(df: pd.DataFrame,
                    n: int = 16,
                    cols: int = 4,
                    label_col: str = "label",
                    title: str = ""):
    """
    Plot a grid of N images drawn from a dataframe that has a 'filepath' column
    and (optionally) a label column.
    """
    assert "filepath" in df.columns, "DataFrame must contain a 'filepath' column."

    paths  = df.sample(n=min(n, len(df)), random_state=42)["filepath"].tolist()
    rows   = (len(paths) + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, p in enumerate(paths, 1):
        img = Image.open(Path(p))
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        if label_col in df.columns:
            lbl = df.loc[df["filepath"] == str(p), label_col].values[0]
            plt.title(str(lbl), fontsize=8)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------

show_df_samples(train_df, n=16, cols=4, label_col="label", title="Train sample")
show_df_samples(test_df,  n=16, cols=4, label_col=None,   title="Test sample")


#Checking the distribution of classes
counts = train_df['label'].value_counts()
print(counts)
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel('Label')
plt.ylabel('Number of Observations')
plt.title('Distribution of Labels in the Training Data')
plt.xticks(ticks=[0, 1], labels=['0: No Cancer', '1: Cancer'])
plt.show()


train_bal = (
    pd.concat([
        train_df.query('label == 1'),
        train_df.query('label == 0').sample(n=len(train_df[train_df['label'] == 1]), random_state=42)
    ])
    .sample(frac=1, random_state=42)      # shuffle
    .reset_index(drop=True)
)


#Checking the distribution of classes
bal_counts = train_bal['label'].value_counts()
print(bal_counts)
sns.barplot(x=bal_counts.index, y=bal_counts.values)
plt.xlabel('Label')
plt.ylabel('Number of Observations')
plt.title('Distribution of Labels in the Balanced Training Data')
plt.xticks(ticks=[0, 1], labels=['0: No Cancer', '1: Cancer'])
plt.show()





# Unified, clean import block
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization, Activation,
    Conv2D, MaxPooling2D, LeakyReLU, SpatialDropout2D
)
from tensorflow.keras import regularizers, layers,models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator 








IMG_SIZE    = (96, 96)
BATCH_SIZE  = 64
VAL_SPLIT   = 0.20
RANDOM_SEED = 42
EPOCHS      = 50               # upper bound, EarlyStopping will stop sooner

bal_train_df, val_df = train_test_split(
    train_bal,
    test_size   = VAL_SPLIT,
    stratify    = train_bal['label'],
    random_state= RANDOM_SEED
)




train_gen = ImageDataGenerator(
    rescale            = 1/255.,
    rotation_range     = 20,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    horizontal_flip    = True,
    vertical_flip      = True,
    zoom_range         = 0.2,
    shear_range        = 0.2,
    fill_mode          = 'nearest'
)
val_gen = ImageDataGenerator(rescale=1/255.)
test_gen = ImageDataGenerator(rescale=1/255.)

train_flow = train_gen.flow_from_dataframe(
    dataframe   = bal_train_df,              
    x_col       = 'filepath',
    y_col       = 'label',
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'raw',
    shuffle     = True,
    validate_filenames=False #Comment out if you want to check file validation
)

val_flow = test_gen.flow_from_dataframe(
    dataframe   = val_df,
    x_col       = 'filepath',
    y_col       = 'label',
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'raw',
    shuffle     = False,
    validate_filenames=False #Comment out if you want to check file validation
)



def build_cnn(num_blocks    = 3,
              start_filters = 32,
              dense_units   = 256,
              dropout_conv  = 0.2,
              dropout_dense = 0.3,
              l2_reg        = 1e-4,
              lr            = 1e-3):
    """
    Build & compile a small-to-medium CNN.
    Args control width/depth so you can grid-search later.
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    x = inputs
    filters = start_filters
    for b in range(num_blocks):
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(dropout_conv)(x)
        filters *= 2                       # double filters each block

    x = layers.Flatten()(x)               # or GlobalAveragePooling2D()
    x = layers.Dense(dense_units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_dense)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer = Adam(learning_rate=lr),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model



# callbacks = [
#     EarlyStopping(patience=3, min_delta=1e-3, restore_best_weights=True, monitor='val_loss'),
#     ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss'),
#     ModelCheckpoint('best_cnn.h5', save_best_only=True, monitor='val_loss')
# ]

# model = build_cnn(num_blocks=3, start_filters=32)

# history = model.fit(
#     train_flow,
#     epochs        = 25,
#     validation_data = val_flow,
#     callbacks     = callbacks,
#     verbose       = 1
# )






import json, os, shutil
from pathlib import Path
import tensorflow as tf

STATE_PATH   = Path("/kaggle/working/train_state.json")

# ðŸ‘‰  point to the file you just added as a dataset input
WEIGHTS_PATH = Path("/kaggle/input/best-cnn-for-cancer-detection/best_cnn.h5")

# â”€â”€â”€ Build model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tf.keras.backend.clear_session()
model = build_cnn(num_blocks=3, start_filters=32)

initial_epoch = 0
if WEIGHTS_PATH.exists():
    model.load_weights(str(WEIGHTS_PATH))

    if STATE_PATH.exists():                 # resume with stored epoch (2nd run+)
        with open(STATE_PATH) as f:
            initial_epoch = json.load(f).get("epoch", 0)
    else:                                   # first run after uploading .h5
        initial_epoch = 12                  # <-- last completed epoch
    print(f"Resuming from epoch {initial_epoch}")

# â”€â”€â”€ Callback that writes train_state.json each epoch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def on_epoch_end(self, epoch, logs=None):
        with open(self.path, "w") as f:
            json.dump({"epoch": epoch + 1}, f)

tracker_cb = EpochTracker(STATE_PATH)

# â”€â”€â”€ Standard callbacks (EarlyStopping, LR plateau, checkpoint) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, min_delta=1e-3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        "/kaggle/working/best_cnn.h5", save_best_only=True, monitor="val_loss"),
    tracker_cb,
]

# â”€â”€â”€ Train (will stop early after â‰¤3 flat epochs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=25,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=1,
)

# â”€â”€â”€ Move large checkpoint out of /kaggle/working before commit packs files â”€â”€â”€â”€
# if Path("/kaggle/working/best_cnn.h5").exists():
#     shutil.move("/kaggle/working/best_cnn.h5", "/kaggle/temp/best_cnn.h5")



plt.plot(history.history['val_auc'], label='val AUC')
plt.plot(history.history['val_accuracy'], label='val Acc')
plt.legend(); plt.title('Validation metrics'); plt.show()



def summarize_run(name, model, history):
    best_idx = np.argmin(history.history['val_loss'])
    return {
        'model'      : name,
        'val_loss'   : history.history['val_loss'][best_idx],
        'val_acc'    : history.history['val_accuracy'][best_idx],
        'val_auc'    : history.history['val_auc'][best_idx],
        'params'     : model.count_params()
    }

results = []
results.append(summarize_run("3-block-32f", model, history))
pd.DataFrame(results)



# --- test generator -------------------------------------------------
test_flow = test_gen.flow_from_dataframe(
    dataframe  = test_df,
    x_col      = "filepath",
    y_col      = None,
    class_mode = None,
    target_size= IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle    = False
)

# --- predict & write submission ------------------------------------
preds = model.predict(test_flow, verbose=1).ravel()
submission = pd.DataFrame({"id": test_df["id"], "label": preds})
submission.to_csv("submission.csv", index=False)
submission.head()







