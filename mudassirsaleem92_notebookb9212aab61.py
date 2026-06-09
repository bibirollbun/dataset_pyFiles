
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


IMG_SIZE = 456
BATCH_SIZE = 2

NUM_CLASSES = 4

MODEL_SEEDS = [42, 123, 777]  
MODEL_DIR = "/kaggle/working/effb3_seeds"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Using model seeds:", MODEL_SEEDS)
print("Models will be saved in:", MODEL_DIR)
print("Image size:", IMG_SIZE, "x", IMG_SIZE, "| Batch size:", BATCH_SIZE)




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




def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # to [0, 1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if label is None:
        return img
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    image = tf.image.random_brightness(image, max_delta=0.20)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    image = tf.image.random_hue(image, max_delta=0.02)

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




FOCAL_GAMMA = 1.5   
FOCAL_ALPHA = 0.7   
FOCAL_LS    = 0.10  

def categorical_focal_loss_with_label_smoothing(gamma=FOCAL_GAMMA,
                                                alpha=FOCAL_ALPHA,
                                                ls=FOCAL_LS,
                                                classes=4):

    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()

        y_pred_ls = (1.0 - ls) * y_pred + ls / classes
        y_pred_ls = tf.clip_by_value(y_pred_ls, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred_ls)

        weight = alpha * y_true * tf.math.pow(1.0 - y_pred_ls, gamma)

        loss = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return tf.reduce_mean(loss)

    return loss_fn

focal_loss = categorical_focal_loss_with_label_smoothing(classes=NUM_CLASSES)

def build_model(seed=None):
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

print("Focal loss config -> gamma:", FOCAL_GAMMA, "| alpha:", FOCAL_ALPHA, "| label_smoothing:", FOCAL_LS)





def create_compiled_model(seed=None):
    model = build_model(seed=seed)

    optimizer = optimizers.Adam(
        learning_rate=2e-4,
        clipnorm=1.0
    )

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

model = create_compiled_model(seed=MODEL_SEEDS[0])

print("Total params:", model.count_params())




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





EPOCHS = 30  

last_history = None

for seed in MODEL_SEEDS:
    weights_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}.weights.h5")
    print("\n==============================")
    print(f"Training model with seed {seed}")
    print("Weights file:", weights_path)

    if os.path.exists(weights_path):
        print("➡ Weights already found on disk. Skipping training for this seed.")
        continue

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





import matplotlib.pyplot as plt

test_paths = test_df["image_path"].values


if "tta_augment_img" not in globals():
    def tta_augment_img(image):

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        return image

if "make_test_dataset" not in globals():
    def make_test_dataset(paths, tta=False):

        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda x: decode_image(x), num_parallel_calls=AUTO)
        if tta:
            ds = ds.map(lambda x: tta_augment_img(x), num_parallel_calls=AUTO)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTO)
        return ds


seed_metrics = []
seed_test_preds_list = []

TTA_ROUNDS_SEEDS = 4

for seed in MODEL_SEEDS:
    weights_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}.weights.h5")
    if not os.path.exists(weights_path):
        print(f"\n❌ Seed {seed}: weights file not found at {weights_path}, skipping this seed.")
        continue

    print("\n==============================")
    print(f"Evaluating and predicting for seed {seed}")
    print("Weights file:", weights_path)

    
    model = create_compiled_model(seed=seed)
    model.load_weights(weights_path)

    results = model.evaluate(val_ds, verbose=0)
    val_loss, val_acc, val_roc_auc = results

    seed_metrics.append({
        "seed": seed,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_roc_auc": val_roc_auc,
    })

    pred_path = os.path.join(MODEL_DIR, f"effb3_seed{seed}_test_preds.npy")
    if os.path.exists(pred_path):
        print(f"Found existing test predictions for seed {seed}, loading from disk...")
        seed_test_preds = np.load(pred_path)
    else:
        print(f"Creating TTA predictions for seed {seed} on the test set...")
        seed_test_preds = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
        for t in range(TTA_ROUNDS_SEEDS):
            use_tta = t > 0  # first pass no TTA, then with TTA
            test_ds_seed = make_test_dataset(test_paths, tta=use_tta)
            preds_t = model.predict(test_ds_seed, verbose=1)
            seed_test_preds += preds_t / TTA_ROUNDS_SEEDS
        np.save(pred_path, seed_test_preds)
        print(f"Saved seed {seed} test predictions to {pred_path}")

    seed_test_preds_list.append(seed_test_preds)

if len(seed_metrics) == 0:
    print("\nNo seed models were evaluated (maybe weights are missing).")
else:
    seed_metrics_df = pd.DataFrame(seed_metrics)
    print("\nSeed validation performance (global 10% val split):")
    print(seed_metrics_df)

    aucs = seed_metrics_df["val_roc_auc"].values
    weights = aucs / aucs.sum()
    seed_metrics_df["weight"] = weights

    print("\nSeed weights based on validation ROC AUC:")
    print(seed_metrics_df)

    seed_preds_stack = np.stack(seed_test_preds_list, axis=0)  # [num_seeds, num_test, num_classes]
    weighted_seed_pred = np.tensordot(weights, seed_preds_stack, axes=(0, 0))

    submission_seed_weighted = pd.DataFrame(weighted_seed_pred, columns=TARGET_COLS)
    submission_seed_weighted.insert(0, "image_id", test_df["image_id"].values)

    seed_ensemble_filename = "submission_effb3_seeds_weighted_tta_b3.csv"
    submission_seed_weighted.to_csv(seed_ensemble_filename, index=False)

    print("\nSaved weighted seed ensemble submission to", seed_ensemble_filename)

    plt.figure(figsize=(6, 4))
    plt.bar(seed_metrics_df["seed"].astype(str), seed_metrics_df["val_roc_auc"])
    for i, (s, auc) in enumerate(zip(seed_metrics_df["seed"], seed_metrics_df["val_roc_auc"])):
        plt.text(i, auc, f"{auc:.4f}", ha="center", va="bottom")
    plt.xlabel("Seed")
    plt.ylabel("Validation ROC AUC")
    plt.title("Validation ROC AUC per seed (global val split)")
    plt.tight_layout()
    plt.show()




from sklearn.model_selection import StratifiedKFold

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




import matplotlib.pyplot as plt

N_FOLDS = 5
EPOCHS_FOLD = 15
TTA_ROUNDS = 4

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_paths = train_df["image_path"].values
all_labels = train_df[TARGET_COLS].values.astype("float32")
all_label_idx = train_df["label_idx"].values

test_paths = test_df["image_path"].values

fold_metrics = []        
fold_preds_list = []     
fold_ids = []            

fold_idx = 0
for train_idx, val_idx in skf.split(all_paths, all_label_idx):
    fold_idx += 1
    print(f"\n========== Fold {fold_idx}/{N_FOLDS} ==========")

    ckpt_name = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}.weights.h5")
    pred_path = os.path.join(MODEL_DIR, f"effb3_fold{fold_idx}_test_preds.npy")

    x_tr, x_val = all_paths[train_idx], all_paths[val_idx]
    y_tr, y_val = all_labels[train_idx], all_labels[val_idx]

    train_ds_fold = make_train_dataset(x_tr, y_tr)
    val_ds_fold = make_val_dataset(x_val, y_val)


    model = create_compiled_model(seed=SEED + fold_idx)

    if os.path.exists(ckpt_name):
        print(f"➡ Found existing weights for fold {fold_idx}, skipping training.")
        model.load_weights(ckpt_name)
    else:
        print("\n--- Training EfficientNetB3 on this fold ---")

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

        model.load_weights(ckpt_name)

    results = model.evaluate(val_ds_fold, verbose=0)
    val_loss, val_acc, val_roc_auc = results

    fold_metrics.append({
        "fold": fold_idx,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_roc_auc": val_roc_auc,
        "n_val_samples": len(x_val),
    })
    fold_ids.append(fold_idx)

    print(f"Fold {fold_idx} validation ROC AUC: {val_roc_auc:.6f}")

    if os.path.exists(pred_path):
        print(f"Found saved test predictions for fold {fold_idx}, loading from disk...")
        fold_test_preds = np.load(pred_path)
    else:
        print(f"Predicting on test set with TTA for fold {fold_idx}...")
        fold_test_preds = np.zeros((len(test_paths), NUM_CLASSES), dtype="float32")
        for t in range(TTA_ROUNDS):
            use_tta = t > 0
            test_ds_fold = make_test_dataset(test_paths, tta=use_tta)
            preds_t = model.predict(test_ds_fold, verbose=1)
            fold_test_preds += preds_t / TTA_ROUNDS
        np.save(pred_path, fold_test_preds)
        print(f"Saved test predictions for fold {fold_idx} to {pred_path}")

    fold_preds_list.append(fold_test_preds)

if len(fold_preds_list) == 0:
    raise RuntimeError("No fold predictions found. Please ensure at least one fold finished.")

fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold")

print("\nPer-fold validation performance:")
print(fold_metrics_df)

final_test_pred_mean = np.mean(np.stack(fold_preds_list, axis=0), axis=0)

submission_kfold_mean = pd.DataFrame(final_test_pred_mean, columns=TARGET_COLS)
submission_kfold_mean.insert(0, "image_id", test_df["image_id"].values)
submission_kfold_mean.to_csv("submission_kfold_tta_b3_resumable.csv", index=False)
print("\nSaved simple mean K-fold + TTA ensemble to submission_kfold_tta_b3_resumable.csv")

aucs = fold_metrics_df["val_roc_auc"].values
weights = aucs / aucs.sum()
fold_metrics_df["weight"] = weights

print("\nPer-fold validation performance with weights:")
print(fold_metrics_df)

fold_preds_stack = np.stack(fold_preds_list, axis=0)  # [num_folds, num_test, num_classes]
final_test_pred_weighted = np.tensordot(weights, fold_preds_stack, axes=(0, 0))

submission_kfold_weighted = pd.DataFrame(final_test_pred_weighted, columns=TARGET_COLS)
submission_kfold_weighted.insert(0, "image_id", test_df["image_id"].values)
submission_kfold_weighted.to_csv("submission_kfold_tta_b3_weighted.csv", index=False)

print("\nSaved weighted K-fold + TTA ensemble to submission_kfold_tta_b3_weighted.csv")

plt.figure(figsize=(6, 4))
plt.bar(fold_metrics_df["fold"].astype(str), fold_metrics_df["val_roc_auc"])
for i, (f, auc) in enumerate(zip(fold_metrics_df["fold"], fold_metrics_df["val_roc_auc"])):
    plt.text(i, auc, f"{auc:.4f}", ha="center", va="bottom")
plt.xlabel("Fold")
plt.ylabel("Validation ROC AUC")
plt.title("Validation ROC AUC per fold")
plt.tight_layout()
plt.show()




if "weighted_seed_pred" not in globals():
    raise RuntimeError("weighted_seed_pred not found. Please run Cell 8 (seed weighted ensemble) first.")

if "seed_metrics_df" not in globals():
    raise RuntimeError("seed_metrics_df not found. Please run Cell 8 (seed metrics) first.")

if "final_test_pred_weighted" not in globals():
    raise RuntimeError("final_test_pred_weighted not found. Please run Cell 10 (weighted K-fold) first.")

if "fold_metrics_df" not in globals():
    raise RuntimeError("fold_metrics_df not found. Please run Cell 10 (fold metrics) first.")

print("\n==============================")
print("Step 1: Using existing weighted seed and K-fold ensembles")

print("\nSeed validation performance (from Cell 8):")
print(seed_metrics_df)

print("\nK-fold validation performance (from Cell 10):")
print(fold_metrics_df)

score_seed = seed_metrics_df["val_roc_auc"].mean()
score_fold = fold_metrics_df["val_roc_auc"].mean()

print(f"\nAverage validation ROC AUC for seed ensemble  : {score_seed:.6f}")
print(f"Average validation ROC AUC for K-fold ensemble: {score_fold:.6f}")


total_score = score_seed + score_fold
w_seed_final = score_seed / total_score
w_fold_final = score_fold / total_score

print(f"\nFinal top-level weights based on mean ROC AUC:")
print(f"  Seed ensemble weight : {w_seed_final:.4f}")
print(f"  K-fold ensemble weight: {w_fold_final:.4f}")



final_ensemble_pred = (
    w_fold_final * final_test_pred_weighted +
    w_seed_final * weighted_seed_pred
)


submission_ensemble = pd.DataFrame(final_ensemble_pred, columns=TARGET_COLS)
submission_ensemble.insert(0, "image_id", test_df["image_id"].values)

ensemble_filename = "submission_effb3_seeds_kfold_tta_ensemble_weighted_mean.csv"
submission_ensemble.to_csv(ensemble_filename, index=False)

print("\nSaved final combined (weighted seeds + weighted K-fold, with data-driven top-level weights) submission to")
print(" ", ensemble_filename)

submission_ensemble.head()


