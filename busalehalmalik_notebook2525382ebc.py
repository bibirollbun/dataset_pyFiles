# Cell 1: Here I am installing all libraries e.g. TensorFlow/Keras, EfficientNetB3,
# also I am defining global variables (here I also increase image size a bit to give more details)

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

# Here I am fixing the random seed so training is more reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTO = tf.data.AUTOTUNE

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"

# ğŸ”¹ Here I am increasing the image size from 320 to 380 so EfficientNetB3 can see more details.
#    Batch size 4 is still safe inside Kaggle GPU limit for this size (if OOM I can reduce to 352 or go back to 320).
IMG_SIZE = 456
BATCH_SIZE = 2

NUM_CLASSES = 4

# ğŸ”¹ Multi-seed ensemble config (I train same B3 model with different seeds)
MODEL_SEEDS = [42, 123, 777]  # I can change or reduce to 2 seeds if GPU is slow
MODEL_DIR = "/kaggle/working/effb3_seeds"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Using model seeds:", MODEL_SEEDS)
print("Models will be saved in:", MODEL_DIR)
print("Image size:", IMG_SIZE, "x", IMG_SIZE, "| Batch size:", BATCH_SIZE)



# Cell 2: In this cell I am defining four target columns, and splitting the dataset to train / validate

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Here I am creating one label index (0..3) from the one-hot targets so I can use it for stratified split
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



# Cell 3: Image resize, Augmentation (flip, rotation, color jitter), Extracts image paths
# Here I am making augmentation a bit stronger so model can generalize better.

def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # to [0, 1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if label is None:
        return img
    return img, label

def augment(image, label):
    # random flips (same as before)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # ğŸ”¹ new: random 90-degree rotation (0, 90, 180, 270)
    # this helps the model see leaves in different orientations
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # ğŸ”¹ new: a bit stronger color jitter (brightness / contrast / saturation / tiny hue)
    image = tf.image.random_brightness(image, max_delta=0.20)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    image = tf.image.random_hue(image, max_delta=0.02)

    return image, label

# Here I am taking the labels and paths for train and validation
train_labels = train_df[TARGET_COLS].values.astype("float32")
val_labels = val_df[TARGET_COLS].values.astype("float32")

train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values

# Here I am building the train dataset, with shuffle + decode + augment + batch + prefetch
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(len(train_paths), seed=SEED)
    .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    .map(augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Validation dataset (only resize, no augmentation)
val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



# Cell 4: Defining focal loss that will focus on hard/rare examples.
# Defining the model (EfficientNet B3, Dropout, output layer with 4-way softmax for the four classes)

# Here I am defining some global hyper parameters for focal loss so I can easily change them later
FOCAL_GAMMA = 1.5   # how strong to focus on hard examples (2.0 is stronger, 1.0 is like normal CE)
FOCAL_ALPHA = 0.7   # class balancing factor (0.5 = equal, >0.5 gives a bit more weight)
FOCAL_LS    = 0.10  # label smoothing strength (0 = no smoothing)

def categorical_focal_loss_with_label_smoothing(gamma=FOCAL_GAMMA,
                                                alpha=FOCAL_ALPHA,
                                                ls=FOCAL_LS,
                                                classes=4):
    """
    Here I am using focal loss with small label smoothing.
    gamma  -> focus on hard samples (bigger gamma = more focus)
    alpha  -> balance positive/negative contribution
    ls     -> label smoothing value to avoid over-confident predictions
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()

        # apply label smoothing on the predictions side so we avoid exact 0/1 logits
        y_pred_ls = (1.0 - ls) * y_pred + ls / classes
        y_pred_ls = tf.clip_by_value(y_pred_ls, epsilon, 1.0 - epsilon)

        # standard cross entropy but using the smoothed predictions
        cross_entropy = -y_true * tf.math.log(y_pred_ls)

        # focal weighting: down-weight easy examples, focus more on hard ones
        weight = alpha * y_true * tf.math.pow(1.0 - y_pred_ls, gamma)

        loss = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return tf.reduce_mean(loss)

    return loss_fn

# Here I am creating one focal loss instance for 4 classes
focal_loss = categorical_focal_loss_with_label_smoothing(classes=NUM_CLASSES)

def build_model(seed=None):
    # ğŸ”¹ NEW: allow per-model seeding for ensemble members
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

# here I can print the focal loss config if I want to double check
print("Focal loss config -> gamma:", FOCAL_GAMMA, "| alpha:", FOCAL_ALPHA, "| label_smoothing:", FOCAL_LS)



# Cell 5: Here I am Building the model, Adam optimizer, compile the model only B3
# I am also adding small gradient clipping to make training more stable.

def create_compiled_model(seed=None):
    # here I am creating the EfficientNetB3 model with the given seed
    model = build_model(seed=seed)

    # Adam optimizer with small learning rate and gradient clipping
    # lr = 2e-4 worked well before, clipnorm=1.0 helps to avoid exploding gradients
    optimizer = optimizers.Adam(
        learning_rate=2e-4,
        clipnorm=1.0
    )

    # ROC AUC metric for 4-way multi-label (one-hot) plant disease classification
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



# Cell 5 extra: Here I will drawing my model so I can see the architecture of EfficientNetB3

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



# Cell 6: Training, Using ModelCheckpoint to save whenever val_roc_auc improve,
# changing the learning rate whenever roc auc stop improving for 2 epchs.
# Also I will do early stop if there is no improvement.
# Here I am training multiple EfficientNetB3 models (different seeds) and skipping if weights already exist.

EPOCHS = 30  # I keep 30 epochs, early stopping will stop before that if no improvement

# store histories for the last trained model (optional for plotting)
last_history = None

for seed in MODEL_SEEDS:
    weights_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}.weights.h5")
    print("\n==============================")
    print(f"Training model with seed {seed}")
    print("Weights file:", weights_path)

    # if this seed already trained before, I skip to save time when kernel restart
    if os.path.exists(weights_path):
        print("â�¡ Weights already found on disk. Skipping training for this seed.")
        continue

    # build & compile model for this seed (uses our focal loss and augmentations)
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
        verbose=1,
    )

    last_history = history  

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



# Cell 9

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
    return image

def make_test_dataset(paths, tta=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda x: decode_image(x), num_parallel_calls=AUTO)
    if tta:
        ds = ds.map(lambda x: tta_augment_img(x), num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds



# Cell 10 â€” K-fold + TTA with EfficientNetB3 only (resumable)

N_FOLDS = 5
EPOCHS_FOLD = 15
TTA_ROUNDS = 4

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_paths = train_df["image_path"].values
all_labels = train_df[TARGET_COLS].values.astype("float32")
all_label_idx = train_df["label_idx"].values

test_paths = test_df["image_path"].values

fold_pred_files = []
for fold in range(1, N_FOLDS + 1):
    fold_pred_files.append(os.path.join(MODEL_DIR, f"effb3_fold{fold}_test_preds.npy"))

completed_folds = []
fold_preds_list = []

# ğŸ”¹ First, load preds for any folds already finished (resume support)
for fold in range(1, N_FOLDS + 1):
    pred_path = os.path.join(MODEL_DIR, f"effb3_fold{fold}_test_preds.npy")
    if os.path.exists(pred_path):
        print(f"Found saved predictions for fold {fold}, loading...")
        fold_preds = np.load(pred_path)
        fold_preds_list.append(fold_preds)
        completed_folds.append(fold)

# ğŸ”¹ Now train / predict remaining folds
fold_idx = 0
for train_idx, val_idx in skf.split(all_paths, all_label_idx):
    fold_idx += 1
    print(f"\n========== Fold {fold_idx}/{N_FOLDS} ==========")

    ckpt_name = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}.weights.h5")
    pred_path = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}_test_preds.npy")

    # If this fold is already done, skip
    if os.path.exists(ckpt_name) and os.path.exists(pred_path):
        print(f"â�¡ Fold {fold_idx} already has weights and predictions, skipping training.")
        continue

    x_tr, x_val = all_paths[train_idx], all_paths[val_idx]
    y_tr, y_val = all_labels[train_idx], all_labels[val_idx]

    train_ds_fold = make_train_dataset(x_tr, y_tr)
    val_ds_fold = make_val_dataset(x_val, y_val)

    print("\n--- Backbone b3 ---")
    model = build_model_backbone(backbone="b3")
    optimizer = optimizers.Adam(learning_rate=1e-4)
    auc_metric = tf.keras.metrics.AUC(
        multi_label=True,
        num_labels=NUM_CLASSES,
        name="roc_auc"
    )
    model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=["accuracy", auc_metric]
    )

    callbacks = [
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

    history_fold = model.fit(
        train_ds_fold,
        validation_data=val_ds_fold,
        epochs=EPOCHS_FOLD,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best weights before inference
    model.load_weights(ckpt_name)

    print(f"Predicting on test set with TTA for fold {fold_idx}...")
    fold_test_preds = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
    for t in range(TTA_ROUNDS):
        use_tta = t > 0
        test_ds_fold = make_test_dataset(test_paths, tta=use_tta)
        preds_t = model.predict(test_ds_fold, verbose=1)
        fold_test_preds += preds_t / TTA_ROUNDS

    # Save fold predictions so we can resume later
    np.save(pred_path, fold_test_preds)
    fold_preds_list.append(fold_test_preds)
    completed_folds.append(fold_idx)

# ğŸ”¹ Combine all available folds into final prediction
if len(fold_preds_list) == 0:
    raise RuntimeError("No fold predictions found. Please ensure at least one fold finished.")

final_test_pred = np.mean(np.stack(fold_preds_list, axis=0), axis=0)

submission_kfold = pd.DataFrame(final_test_pred, columns=TARGET_COLS)
submission_kfold.insert(0, "image_id", test_df["image_id"].values)
submission_kfold.to_csv("submission_kfold_tta_b3_resumable.csv", index=False)

print("Completed folds:", completed_folds)
print("Saved K-fold + TTA ensemble submission to submission_kfold_tta_b3_resumable.csv")

submission_kfold.head()



# Cell 11: Final ensemble using 3 seed models (Cell 6) + 5-fold models (Cell 10) with TTA.
# Here I am trying 50/50 weighting between K-fold ensemble and seed ensemble.

# I will re-use the test paths
test_paths = test_df["image_path"].values

# 1) Get predictions from the 3 seed models (global training from Cell 6)
seed_test_preds_list = []

print("\n==============================")
print("Getting TTA predictions from seed models (Cell 6 weights)...")

for seed in MODEL_SEEDS:
    weights_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}.weights.h5")
    if not os.path.exists(weights_path):
        print(f"â�Œ Seed {seed}: weights file not found at {weights_path}, skipping this seed.")
        continue

    print(f"\nSeed {seed}: loading model from {weights_path}")
    model = create_compiled_model(seed=seed)
    model.load_weights(weights_path)

    # Here I am doing the same style of TTA as in K-fold (TTA_ROUNDS from Cell 10)
    seed_preds = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
    for t in range(TTA_ROUNDS):
        use_tta = t > 0  # first pass no TTA, then with TTA
        test_ds_seed = make_test_dataset(test_paths, tta=use_tta)
        preds_t = model.predict(test_ds_seed, verbose=1)
        seed_preds += preds_t / TTA_ROUNDS

    seed_test_preds_list.append(seed_preds)

if len(seed_test_preds_list) > 0:
    seed_mean_pred = np.mean(np.stack(seed_test_preds_list, axis=0), axis=0)
    print("\nFinished seed-model TTA predictions.")
else:
    seed_mean_pred = None
    print("\nNo seed-model predictions were created (maybe weights missing).")

# 2) Load predictions from the 5 K-fold models (already saved in Cell 10)
fold_test_preds_list = []

print("\n==============================")
print("Loading saved K-fold predictions from disk...")

for fold in range(1, N_FOLDS + 1):
    pred_path = os.path.join(MODEL_DIR, f"effb3_fold{fold}_test_preds.npy")
    if os.path.exists(pred_path):
        print(f"Fold {fold}: loading predictions from {pred_path}")
        fold_preds = np.load(pred_path)
        fold_test_preds_list.append(fold_preds)
    else:
        print(f"â�Œ Fold {fold}: prediction file not found at {pred_path}, skipping this fold.")

if len(fold_test_preds_list) == 0:
    raise RuntimeError("No fold predictions found. Please make sure Cell 10 finished correctly.")

fold_mean_pred = np.mean(np.stack(fold_test_preds_list, axis=0), axis=0)
print("Finished loading K-fold predictions.")

# 3) Combine seed ensemble and K-fold ensemble
print("\n==============================")
print("Combining seed ensemble + K-fold ensemble into final prediction...")

if seed_mean_pred is not None:
    # ğŸ”¹ 50/50 weighting between K-fold and seeds
    FINAL_FOLD_WEIGHT = 0.5
    FINAL_SEED_WEIGHT = 0.5

    final_ensemble_pred = (
        FINAL_FOLD_WEIGHT * fold_mean_pred +
        FINAL_SEED_WEIGHT * seed_mean_pred
    )

    print(f"Using weighted ensemble: {FINAL_FOLD_WEIGHT:.2f} * K-fold + {FINAL_SEED_WEIGHT:.2f} * seeds")
else:
    # fallback: only K-fold predictions if no seed models available
    final_ensemble_pred = fold_mean_pred
    print("No seed predictions available, using K-fold ensemble only.")

# 4) Create final submission file for ensemble
submission_ensemble = pd.DataFrame(final_ensemble_pred, columns=TARGET_COLS)
submission_ensemble.insert(0, "image_id", test_df["image_id"].values)

ensemble_filename = "submission_effb3_seeds_kfold_tta_ensemble_w50_50.csv"
submission_ensemble.to_csv(ensemble_filename, index=False)

print("\nSaved final ensemble submission to", ensemble_filename)
submission_ensemble.head()


