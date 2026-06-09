# Cell 1: Here I am installing all libraries e.g. TensorFlow/Keras, EfficientNetB3, also I am defining global variables

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTO = tf.data.AUTOTUNE

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_SIZE = 380          # increased from 320 to capture more fine-grained leaf details
BATCH_SIZE = 2          # can reduce to 2 if GPU runs out of memory
NUM_CLASSES = 4

# multi-seed ensemble config for EfficientNetB3 single-split training
MODEL_SEEDS = [42, 123, 777]
MODEL_DIR = "/kaggle/working/effb3_seeds"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Using model seeds:", MODEL_SEEDS)
print("Models will be saved in:", MODEL_DIR)



# Cell 2: In this cell I am defining four target columns, and splitting the dataset to train / validate

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

train_df["label_idx"] = train_df[TARGET_COLS].values.argmax(axis=1)

train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(DATA_DIR, "images", f"{x}.jpg")
)
test_df["image_path"] = test_df["image_id"].apply(
    lambda x: os.path.join(DATA_DIR, "images", f"{x}.jpg")
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label_idx"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))

train_df.head()



# Cell 3: Image resize, Augmentation (flips, brightness, contrast, random saturation), Extracts image paths


def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if label is None:
        return img
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # extra color jitter to improve robustness
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label

train_labels = train_df[TARGET_COLS].values.astype("float32")
val_labels = val_df[TARGET_COLS].values.astype("float32")

train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(len(train_paths), seed=SEED)
    .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    .map(augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



# Cell 4: Defining focal loss that will focus on hard/rare examples.
# Defining the model (EfficientNet B3, Dropout, output layer with 4-way softmax for the four classes)
# Cell 4: Defining focal loss and the EfficientNetB3 model architecture with 4-way softmax output

def categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.75, ls=0.125, classes=4):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred_ls = (1 - ls) * y_pred + ls / classes
        y_pred_ls = tf.clip_by_value(y_pred_ls, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred_ls)
        weight = alpha * y_true * tf.math.pow(1 - y_pred_ls, gamma)
        loss = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return tf.reduce_mean(loss)
    return loss_fn

def build_model(seed=None):
    # allow per-model seeding for the single-split B3 ensembles
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
    x = layers.Dropout(0.3)(base.output)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# slightly tuned focal loss hyperparameters for more stable training
focal_loss = categorical_focal_loss_with_label_smoothing(
    gamma=1.5,
    alpha=0.8,
    ls=0.125,
    classes=NUM_CLASSES,
)



# Cell 5: Here I am Building the model, Adam optimizer, compile the model  

def create_compiled_model(seed=None):
    model = build_model(seed=seed)

    optimizer = optimizers.Adam(learning_rate=2e-4)

    roc_auc_metric = tf.keras.metrics.AUC(
        multi_label=True,
        num_labels=NUM_CLASSES,
        name="roc_auc"
    )

    model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=["accuracy", roc_auc_metric]
    )
    return model

# Build one reference model (first seed) for summary and diagram
model = create_compiled_model(seed=MODEL_SEEDS[0])

print("Total params:", model.count_params())



# Cell 5 extra: Here I will drawing my model

!pip install -q pydot graphviz

from tensorflow.keras.utils import plot_model

plot_model(
    model,
    to_file="efficientnet_b3_model.png",
    show_shapes=True,
    show_layer_names=False,
    dpi=96
)

print("Saved diagram as efficientnet_b3_model.png")



# Cell 6: Training, Using modelchekpoint to save whenever roc improve, changing the learning rate whenever roc auc stop improving for 2 epchs.
# Also I will do early stop if there is no improvement.
# Updated to train multiple EfficientNetB3 models with different seeds and resume if weights exist.

EPOCHS = 30

# store histories for the last trained model (optional for plotting)
last_history = None

for seed in MODEL_SEEDS:
    weights_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}.weights.h5")
    print(f"\n==============================")
    print(f"Training model with seed {seed}")
    print("Weights file:", weights_path)

    if os.path.exists(weights_path):
        print("➡ Weights already found on disk. Skipping training for this seed.")
        continue

    # build & compile model for this seed
    model = create_compiled_model(seed=seed)

    callbacks = [
        ModelCheckpoint(
            weights_path,
            monitor="val_roc_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_roc_auc",
            patience=6,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    last_history = history  # keep the last history for plotting if needed

print("\nAll requested seeds processed. If the kernel dies, rerun up to this cell and any already-trained seeds will be skipped.")



# Cell 7

import matplotlib.pyplot as plt

if "history" not in globals() and "last_history" not in globals():
    print("No training history available in this run (all models were loaded from disk). Skipping plots.")
else:
    # use last_history if available, otherwise the old 'history'
    history_obj = last_history if "last_history" in globals() and last_history is not None else history
    history_dict = history_obj.history

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict["accuracy"], label="train_acc")
    plt.plot(history_dict["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history_dict["roc_auc"], label="train_roc_auc")
    plt.plot(history_dict["val_roc_auc"], label="val_roc_auc")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.title("ROC AUC")

    plt.tight_layout()
    plt.show()



# Cell 8: Here I am training a single EfficientNetB4 model on the same train/validation split to compare with B3

from tensorflow.keras.applications import EfficientNetB4

def build_b4_single_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    x = layers.Dropout(0.3)(base.output)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

b4_single_weights_path = os.path.join(MODEL_DIR, "effb4_single_split.weights.h5")

if os.path.exists(b4_single_weights_path):
    print("Found existing EfficientNetB4 single-split weights, loading from disk...")
    b4_single_model = build_b4_single_model()
    b4_single_model.load_weights(b4_single_weights_path)
else:
    print("Training EfficientNetB4 single-split model on train/validation split...")

    b4_single_model = build_b4_single_model()
    optimizer_b4 = optimizers.Adam(learning_rate=1e-4)
    auc_metric_b4 = tf.keras.metrics.AUC(
        multi_label=True,
        num_labels=NUM_CLASSES,
        name="roc_auc"
    )

    b4_single_model.compile(
        optimizer=optimizer_b4,
        loss=focal_loss,
        metrics=["accuracy", auc_metric_b4],
    )

    callbacks_b4_single = [
        ModelCheckpoint(
            b4_single_weights_path,
            monitor="val_roc_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_roc_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history_b4_single = b4_single_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks_b4_single,
        verbose=1,
    )

    print("Finished training EfficientNetB4 single-split model. Best weights saved at:", b4_single_weights_path)



# Cell 9: Here I define shared helper functions for K-fold training (B3 and B4), including model backbones and TTA datasets

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import EfficientNetB4

def build_model_backbone(backbone="b3"):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    if backbone == "b3":
        base = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling="avg",
        )
    elif backbone == "b4":
        base = EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    x = layers.Dropout(0.3)(base.output)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def make_train_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    ds = ds.map(augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds

def make_val_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds

def tta_augment_img(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    return image

def make_test_dataset(paths, tta=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda x: decode_image(x), num_parallel_calls=AUTO)
    if tta:
        ds = ds.map(lambda x: tta_augment_img(x), num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds



# Cell 10: Here I perform Stratified K-fold training with EfficientNetB3 only and save test predictions with TTA per fold

N_FOLDS = 5
EPOCHS_FOLD = 15
TTA_ROUNDS = 4

skf_b3 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_paths = train_df["image_path"].values
all_labels = train_df[TARGET_COLS].values.astype("float32")
all_label_idx = train_df["label_idx"].values

test_paths = test_df["image_path"].values

fold_preds_b3 = []
completed_folds_b3 = []

for fold_idx, (train_idx, val_idx) in enumerate(skf_b3.split(all_paths, all_label_idx), start=1):
    print(f"\n========== B3 Fold {fold_idx}/{N_FOLDS} ==========")

    ckpt_name = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}.weights.h5")
    pred_path = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}_test_preds.npy")

    if os.path.exists(ckpt_name) and os.path.exists(pred_path):
        print(f"➡ B3 fold {fold_idx} already has weights and predictions, loading predictions and skipping training.")
        fold_preds_b3.append(np.load(pred_path))
        completed_folds_b3.append(fold_idx)
        continue

    x_tr, x_val = all_paths[train_idx], all_paths[val_idx]
    y_tr, y_val = all_labels[train_idx], all_labels[val_idx]

    train_ds_fold = make_train_dataset(x_tr, y_tr)
    val_ds_fold = make_val_dataset(x_val, y_val)

    print("\n--- B3 backbone ---")
    model_b3_fold = build_model_backbone(backbone="b3")
    optimizer_b3_fold = optimizers.Adam(learning_rate=1e-4)
    auc_metric_b3_fold = tf.keras.metrics.AUC(
        multi_label=True,
        num_labels=NUM_CLASSES,
        name="roc_auc"
    )

    model_b3_fold.compile(
        optimizer=optimizer_b3_fold,
        loss=focal_loss,
        metrics=["accuracy", auc_metric_b3_fold]
    )

    callbacks_b3_fold = [
        ModelCheckpoint(
            ckpt_name,
            monitor="val_roc_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_roc_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history_b3_fold = model_b3_fold.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=EPOCHS_FOLD,
        callbacks=callbacks_b3_fold,
        verbose=1,
    )

    model_b3_fold.load_weights(ckpt_name)

    print(f"Predicting on test set with B3 + TTA for fold {fold_idx}...")
    fold_test_preds_b3 = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
    for t in range(TTA_ROUNDS):
        use_tta = t > 0
        test_ds_b3_fold = make_test_dataset(test_paths, tta=use_tta)
        preds_t = model_b3_fold.predict(test_ds_b3_fold, verbose=1)
        fold_test_preds_b3 += preds_t / TTA_ROUNDS

    np.save(pred_path, fold_test_preds_b3)
    fold_preds_b3.append(fold_test_preds_b3)
    completed_folds_b3.append(fold_idx)

if len(fold_preds_b3) == 0:
    raise RuntimeError("No EfficientNetB3 fold predictions found. Please ensure at least one fold finished.")

b3_ensemble_pred = np.mean(np.stack(fold_preds_b3, axis=0), axis=0)

submission_b3_kfold = pd.DataFrame(b3_ensemble_pred, columns=TARGET_COLS)
submission_b3_kfold.insert(0, "image_id", test_df["image_id"].values)
submission_b3_kfold.to_csv("submission_kfold_tta_b3_resumable.csv", index=False)

print("Completed B3 folds:", completed_folds_b3)
print("Saved B3 K-fold + TTA ensemble submission to submission_kfold_tta_b3_resumable.csv")
submission_b3_kfold.head()



# Cell 11: Here I perform a lighter Stratified K-fold training with EfficientNetB4
# to save time on Kaggle (3 folds, 10 epochs, 2 TTA rounds).

B4_N_FOLDS = 3        # only 3 folds for B4 (B3 can still use 5)
B4_EPOCHS_FOLD = 10   # fewer epochs than B3
B4_TTA_ROUNDS = 2     # fewer TTA rounds to speed up inference

skf_b4 = StratifiedKFold(n_splits=B4_N_FOLDS, shuffle=True, random_state=SEED)

all_paths = train_df["image_path"].values
all_labels = train_df[TARGET_COLS].values.astype("float32")
all_label_idx = train_df["label_idx"].values

test_paths = test_df["image_path"].values

fold_preds_b4 = []
completed_folds_b4 = []

for fold_idx, (train_idx, val_idx) in enumerate(skf_b4.split(all_paths, all_label_idx), start=1):
    print(f"\n========== B4 Fold {fold_idx}/{B4_N_FOLDS} ==========")

    ckpt_name = os.path.join(MODEL_DIR, f"effb4_fold{fold_idx}.weights.h5")
    pred_path = os.path.join(MODEL_DIR, f"effb4_fold{fold_idx}_test_preds.npy")

    if os.path.exists(ckpt_name) and os.path.exists(pred_path):
        print(f"➡ B4 fold {fold_idx} already has weights and predictions, loading predictions and skipping training.")
        fold_preds_b4.append(np.load(pred_path))
        completed_folds_b4.append(fold_idx)
        continue

    x_tr, x_val = all_paths[train_idx], all_paths[val_idx]
    y_tr, y_val = all_labels[train_idx], all_labels[val_idx]

    train_ds_fold = make_train_dataset(x_tr, y_tr)
    val_ds_fold = make_val_dataset(x_val, y_val)

    print("\n--- B4 backbone ---")
    model_b4_fold = build_model_backbone(backbone="b4")
    optimizer_b4_fold = optimizers.Adam(learning_rate=1e-4)
    auc_metric_b4_fold = tf.keras.metrics.AUC(
        multi_label=True,
        num_labels=NUM_CLASSES,
        name="roc_auc"
    )

    model_b4_fold.compile(
        optimizer=optimizer_b4_fold,
        loss=focal_loss,
        metrics=["accuracy", auc_metric_b4_fold]
    )

    callbacks_b4_fold = [
        ModelCheckpoint(
            ckpt_name,
            monitor="val_roc_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_roc_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history_b4_fold = model_b4_fold.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=B4_EPOCHS_FOLD,
        callbacks=callbacks_b4_fold,
        verbose=1,
    )

    model_b4_fold.load_weights(ckpt_name)

    print(f"Predicting on test set with B4 + TTA for fold {fold_idx}...")
    fold_test_preds_b4 = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
    for t in range(B4_TTA_ROUNDS):
        use_tta = t > 0
        test_ds_b4_fold = make_test_dataset(test_paths, tta=use_tta)
        preds_t = model_b4_fold.predict(test_ds_b4_fold, verbose=1)
        fold_test_preds_b4 += preds_t / B4_TTA_ROUNDS

    np.save(pred_path, fold_test_preds_b4)
    fold_preds_b4.append(fold_test_preds_b4)
    completed_folds_b4.append(fold_idx)

if len(fold_preds_b4) == 0:
    raise RuntimeError("No EfficientNetB4 fold predictions found. Please ensure at least one fold finished.")

b4_ensemble_pred = np.mean(np.stack(fold_preds_b4, axis=0), axis=0)

submission_b4_kfold = pd.DataFrame(b4_ensemble_pred, columns=TARGET_COLS)
submission_b4_kfold.insert(0, "image_id", test_df["image_id"].values)
submission_b4_kfold.to_csv("submission_kfold_tta_b4.csv", index=False)

print("Completed B4 folds:", completed_folds_b4)
print("Saved B4 K-fold + TTA ensemble submission to submission_kfold_tta_b4.csv")
submission_b4_kfold.head()



# Cell 12: Here I ensemble the B3 and B4 K-fold predictions by averaging their probabilities and create the final submission

# 1) Load B3 K-fold ensemble prediction (or recompute from saved folds)
b3_fold_preds_loaded = []
used_b3_folds = []
for fold_idx in range(1, N_FOLDS + 1):
    pred_path_b3 = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}_test_preds.npy")
    if os.path.exists(pred_path_b3):
        b3_fold_preds_loaded.append(np.load(pred_path_b3))
        used_b3_folds.append(fold_idx)

if len(b3_fold_preds_loaded) == 0:
    raise RuntimeError("No B3 K-fold prediction files found. Please run Cell 10 first.")

b3_ensemble_pred = np.mean(np.stack(b3_fold_preds_loaded, axis=0), axis=0)
print("Using B3 folds:", used_b3_folds)

# 2) Load B4 K-fold ensemble prediction
b4_fold_preds_loaded = []
used_b4_folds = []
for fold_idx in range(1, N_FOLDS + 1):
    pred_path_b4 = os.path.join(MODEL_DIR, f"effb4_fold{fold_idx}_test_preds.npy")
    if os.path.exists(pred_path_b4):
        b4_fold_preds_loaded.append(np.load(pred_path_b4))
        used_b4_folds.append(fold_idx)

if len(b4_fold_preds_loaded) == 0:
    raise RuntimeError("No B4 K-fold prediction files found. Please run Cell 11 first.")

b4_ensemble_pred = np.mean(np.stack(b4_fold_preds_loaded, axis=0), axis=0)
print("Using B4 folds:", used_b4_folds)

# 3) Sanity check shapes
if b3_ensemble_pred.shape != b4_ensemble_pred.shape:
    raise RuntimeError(
        f"Shape mismatch between B3 ensemble {b3_ensemble_pred.shape} and B4 ensemble {b4_ensemble_pred.shape}"
    )

# 4) Final ensemble: simple average between B3 and B4 K-fold ensembles
final_ensemble_pred = 0.5 * b3_ensemble_pred + 0.5 * b4_ensemble_pred

submission_b3b4_ensemble = pd.DataFrame(final_ensemble_pred, columns=TARGET_COLS)
submission_b3b4_ensemble.insert(0, "image_id", test_df["image_id"].values)
submission_b3b4_ensemble.to_csv("submission_b3_b4_kfold_ensemble.csv", index=False)

print("Saved final B3+B4 K-fold ensemble submission to submission_b3_b4_kfold_ensemble.csv")
submission_b3b4_ensemble.head()


