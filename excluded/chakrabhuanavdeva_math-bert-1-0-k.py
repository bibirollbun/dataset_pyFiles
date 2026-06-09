# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -----------------------------
# FINAL STABLE TRAINING SCRIPT
# -----------------------------
# Run on Kaggle (preset BERT assets provided in /kaggle/input/bert/...)
# Usage:
#  - DEBUG=True untuk run singkat (1 epoch, tiny subset)
#  - Sesuaikan GLOBAL_BATCH_SIZE / LEARNING_RATE sesuai quota

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_nlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Config
# ---------------------------
DEBUG = False               # True = quick check (low cost)
PRESET = "bert_base_en_uncased"
SEQUENCE_LENGTH = 256
GLOBAL_BATCH_SIZE = 8      # set to 4 if memory issues
LEARNING_RATE = 1e-5
EPOCHS = 10 if not DEBUG else 1
CHECKPOINT_PATH = "checkpoint_best.weights.h5"
FINAL_WEIGHTS = "final_model.weights.h5"
RANDOM_STATE = 42

TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'

# ---------------------------
# Robust label parsing
# ---------------------------
def parse_multilabel_field(cat, miscon):
    if pd.isna(cat) and pd.isna(miscon):
        return []
    cat_s = str(cat).strip() if pd.notna(cat) else ""
    miscon_s = str(miscon).strip() if pd.notna(miscon) else ""
    # If Category column already contains multiple "X:Y" tokens separated by spaces:
    if cat_s != "" and ":" in cat_s and " " in cat_s:
        labels = [tok.strip() for tok in cat_s.split() if ":" in tok]
        return labels
    token = f"{cat_s}:{miscon_s}" if cat_s != "" else (miscon_s or "")
    return [token] if token != "" else []

# ---------------------------
# Load CSV
# ---------------------------
print("Loading CSV...")
df = pd.read_csv(TRAIN_PATH)

# Build text field
df['QuestionText'] = df['QuestionText'].fillna('')
df['MC_Answer'] = df['MC_Answer'].fillna('')
df['StudentExplanation'] = df['StudentExplanation'].fillna('')
df['combined_text'] = df['QuestionText'] + " [SEP] " + df['MC_Answer'] + " [SEP] " + df['StudentExplanation']

# Parse labels: supports either (Category, Misconception) or combined 'Category:Misconception'
if ('Category' in df.columns) and ('Misconception' in df.columns):
    df['labels_list'] = df.apply(lambda r: parse_multilabel_field(r['Category'], r['Misconception']), axis=1)
elif 'Category:Misconception' in df.columns:
    df['labels_list'] = df['Category:Misconception'].fillna('').apply(lambda s: [t for t in str(s).split() if ':' in t])
else:
    raise ValueError("CSV must contain 'Category'+'Misconception' or 'Category:Misconception' column.")

# Drop rows without labels (safe choice)
num_empty = (df['labels_list'].map(len) == 0).sum()
print(f"Rows with 0 parsed labels: {num_empty}")
if num_empty > 0:
    df = df[df['labels_list'].map(len) > 0].reset_index(drop=True)
    print(f"Dropped {num_empty} rows. New size: {len(df)}")

# Binarize multi-labels
mlb = MultiLabelBinarizer()
y_all = mlb.fit_transform(df['labels_list'])
num_classes = len(mlb.classes_)
print("Num classes:", num_classes)
print("Example classes:", mlb.classes_[:10])

# ---------------------------
# Train/Val split
# ---------------------------
X = df['combined_text'].tolist()
y = y_all
X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

if DEBUG:
    print("DEBUG mode: using tiny subset for quick validation")
    k_train = min(256, max(32, int(len(X_train_text) * 0.02)))
    k_val = min(128, max(32, int(len(X_val_text) * 0.02)))
    X_train_text = X_train_text[:k_train]
    y_train = y_train[:k_train]
    X_val_text = X_val_text[:k_val]
    y_val = y_val[:k_val]

# Basic label sanity
if np.isnan(y_train).any() or np.isnan(y_val).any():
    raise ValueError("Found NaN in label arrays (y_train/y_val). Fix parsing.")

# ---------------------------
# Tokenization (preprocessor)
# ---------------------------
print("Loading BertPreprocessor and tokenizing (this may take a bit)...")
preprocessor = keras_nlp.models.BertPreprocessor.from_preset(PRESET, sequence_length=SEQUENCE_LENGTH)
X_train_processed = preprocessor(X_train_text)
X_val_processed = preprocessor(X_val_text)

print("Preprocessing done. Inspecting types:")
for k, v in X_train_processed.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# ---------------------------
# Dataset builder: cast mask to float32 (0/1)
# ---------------------------
def normalize_inputs(x, y):
    token_ids = tf.cast(x['token_ids'], tf.int32)
    segment_ids = tf.cast(x['segment_ids'], tf.int32)
    # Cast padding_mask to float32 (0.0 or 1.0) — avoids bool issues in numeric ops
    padding_mask = tf.cast(x['padding_mask'], tf.float32)
    return {'token_ids': token_ids, 'segment_ids': segment_ids, 'padding_mask': padding_mask}, y

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_processed, y_train))
train_dataset = train_dataset.map(normalize_inputs, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2048, seed=RANDOM_STATE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_processed, y_val))
val_dataset = val_dataset.map(normalize_inputs, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# quick peek at a batch
for xb, yb in train_dataset.take(1):
    print("Batch example dtypes/shapes:")
    print({k: (v.shape, v.dtype) for k, v in xb.items()})
    print("y batch shape:", yb.shape)

# ---------------------------
# Model build (MirroredStrategy if available)
# ---------------------------
try:
    strategy = tf.distribute.MirroredStrategy()
    print("Using MirroredStrategy with replicas:", strategy.num_replicas_in_sync)
except Exception as e:
    print("MirroredStrategy not used:", e)
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    backbone = keras_nlp.models.BertBackbone.from_preset(PRESET)

    token_ids = Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="token_ids")
    segment_ids = Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="segment_ids")
    padding_mask = Input(shape=(SEQUENCE_LENGTH,), dtype=tf.float32, name="padding_mask")  # float32 mask

    bert_out = backbone({"token_ids": token_ids, "padding_mask": padding_mask, "segment_ids": segment_ids})
    sequence_output = bert_out["sequence_output"]  # (batch, seq_len, hidden_dim)

    # CNN head
    convs = []
    for k in [3, 4, 5]:
        c = Conv1D(filters=128, kernel_size=k, activation='relu')(sequence_output)
        p = GlobalMaxPooling1D()(c)
        convs.append(p)
    merged = Concatenate()(convs)
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dropout = Dropout(0.6)(dense1)
    outputs = Dense(num_classes, activation='sigmoid')(dropout)

    model = Model(inputs={"token_ids": token_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}, outputs=outputs)

    # Stage 1: freeze backbone for stable head training
    backbone.trainable = False

    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

model.summary()

# ---------------------------
# Callbacks: checkpoint + early stopping
# ---------------------------
checkpoint_cb = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)
earlystop_cb = EarlyStopping(monitor="val_auc", patience=2, mode="max", restore_best_weights=True, verbose=1)

# ---------------------------
# Train stage 1 (head only)
# ---------------------------
print("Training stage 1 (head only, backbone frozen)")
history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

# ---------------------------
# Optional stage 2: unfreeze part of backbone and fine-tune
# ---------------------------
UNFREEZE = True and not DEBUG
if UNFREEZE:
    print("Preparing stage 2: unfreezing part of backbone for fine-tune")
    # Try to unfreeze a small portion to avoid instability
    try:
        # Default: unfreeze entire backbone's last encoder block(s) if accessible
        # Fallback: unfreeze whole backbone
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
            total_blocks = len(backbone.encoder.layers)
            n_unfreeze = max(1, total_blocks // 6)
            for layer in backbone.encoder.layers[-n_unfreeze:]:
                layer.trainable = True
            print(f"Unfroze last {n_unfreeze} encoder blocks (out of {total_blocks}).")
        else:
            backbone.trainable = True
            print("Backbone encoder structure not found; unfreezing full backbone.")
    except Exception as e:
        print("Unfreeze attempt failed; unfreezing entire backbone. Err:", e)
        backbone.trainable = True

    # Recompile with smaller LR
    with strategy.scope():
        optimizer_ft = Adam(learning_rate=LEARNING_RATE * 0.5, clipnorm=1.0)
        model.compile(optimizer=optimizer_ft, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )

# ---------------------------
# Save final weights (.weights.h5)
# ---------------------------
if os.path.exists(CHECKPOINT_PATH):
    print("Loading best checkpoint and saving to final weights file...")
    model.load_weights(CHECKPOINT_PATH)
    model.save_weights(FINAL_WEIGHTS)
    print(f"Saved best model weights to {FINAL_WEIGHTS}")
else:
    model.save_weights(FINAL_WEIGHTS)
    print(f"Saved current model weights to {FINAL_WEIGHTS}")

print("TRAINING COMPLETE.")





