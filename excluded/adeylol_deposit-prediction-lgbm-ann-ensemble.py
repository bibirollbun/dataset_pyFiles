import tensorflow as tf

# Check available devices
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())



import pandas as pd
from datetime import datetime

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# test data
print(test.shape)
test


# train data
print(train.shape)
train


# Separate numerical and categorical columns

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = train.select_dtypes(include=["object"]).columns

print("Numerical columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))



# Distributions of numerical columns

import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train[col].hist(bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



# Detect & Replace Outliers (Z-score method)

import numpy as np
from scipy import stats

# Copy dataset to avoid modifying original
train_clean = train.copy()

# Exclude 'id' column (not a feature)
num_cols = [c for c in num_cols if c != "id"]

# Set Z-score threshold
threshold = 3

for col in num_cols:
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(train_clean[col]))

    # Print how many outliers detected
    print(f"{col}: {np.sum(z_scores > threshold)} outliers detected")

    # Compute boundaries (mean Â± 3*std)
    mean, std = train_clean[col].mean(), train_clean[col].std()
    upper, lower = mean + threshold * std, mean - threshold * std

    # Replace values outside boundaries with caps
    train_clean[col] = np.where(
        train_clean[col] > upper, upper,
        np.where(train_clean[col] < lower, lower, train_clean[col])
    )

print("\nâœ… Outliers replaced with boundary values (capped).")



# Plot Distributions After Outlier Handling
import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure(figsize=(6,4))
    train_clean[col].hist(bins=50)
    plt.title(f"Distribution of {col} (after outlier handling)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



# Categorical value counts

for col in cat_cols:
    plt.figure(figsize=(6,4))
    train[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()



# category info
for col in cat_cols:
    print(f"\n---- {col} ----")
    print(train[col].value_counts())



# Mixed Binary + One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Columns that are strictly binary â†’ map directly
binary_cols = ["default", "housing", "loan"]

# for col in binary_cols:
#     train[col] = train[col].map({"no": 0, "yes": 1})
for col in binary_cols:
    train[col] = (
        train[col]
        .astype(str)          # convert to string
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # make lowercase
        .map({"no": 0, "yes": 1})
    )


# Columns that need one-hot encoding
onehot_cols = ["job", "marital", "education", "contact", "month", "poutcome"]

# Apply One-Hot
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(train[onehot_cols])

print("One-hot encoded categorical shape:", X_cat.shape)



print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())

print(train[binary_cols].nunique())
print(train[binary_cols].isna().sum())
print(train[binary_cols].head(10))


# Numerical + Target
from sklearn.preprocessing import StandardScaler
import numpy as np

# Exclude id + target
num_features = [c for c in num_cols if c not in ["id", "y"]]

# Scale numerical features
scaler = StandardScaler()
X_num = scaler.fit_transform(train[num_features])

# Combine numerical + categorical
X = np.hstack([X_num, X_cat, train[binary_cols].values])

# Target variable
y = train["y"].values

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)



# Final updated dataframe for training


# Get names for one-hot encoded columns
onehot_feature_names = encoder.get_feature_names_out(onehot_cols)

# Combine all features into a DataFrame
train_updated = pd.DataFrame(
    np.hstack([X_num, X_cat, train[binary_cols].values]),
    columns=list(num_features) + list(onehot_feature_names) + binary_cols
)

# Add target column
train_updated["y"] = y

print("Final dataframe shape:", train_updated.shape)
train_updated.head()


# split + class weights

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_val.shape)

# class weights to help with imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
class_weight_dict




import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# --- Build test matrix using the SAME encoder/scaler/binary mapping ---
# assumes you already have:
# - train, test
# - binary_cols, onehot_cols, num_features
# - encoder (fitted), scaler (fitted)
# - X, y, X_train, X_val, y_train, y_val
# - class_weight_dict

# map binary in test exactly like train
for col in binary_cols:
    test[col] = (
        test[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

X_num_test = scaler.transform(test[num_features])
X_cat_test = encoder.transform(test[onehot_cols])
X_test = np.hstack([X_num_test, X_cat_test, test[binary_cols].values])




# --- LGBM model ---
# lgbm = lgb.LGBMClassifier(
#         n_estimators=38000, # 40000 was best
#         class_weight='balanced',   # keep balanced handling
#         learning_rate=0.066, # 0.65 was best
#         num_leaves=128,
#         max_depth=12,
#         min_child_samples=12,
#         subsample=0.8,
#         colsample_bytree=0.5,
#         reg_alpha=1.0,
#         reg_lambda=0.3,
#         max_bin=4900,
#         random_state=2003,
#         boosting_type='gbdt',
#         metric='auc',
#         verbosity=-1   # keep model internal logs silent, use callback instead
#         # device='gpu' # enable GPU
# )

lgbm = lgb.LGBMClassifier(
        n_estimators=44000,
        class_weight='balanced',   # keep balanced handling
        learning_rate=0.069, #0.065
        num_leaves=128,
        max_depth=12,
        min_child_samples=12,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=0.3,
        max_bin=5000,
        random_state=2003,
        boosting_type='gbdt',
        metric='auc',
        verbosity=-1
        # device='gpu' # enable GPU
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=300, verbose=True),
        lgb.log_evaluation(period=100)   # print eval every 100 iterations
    ]
)

val_pred_lgbm = lgbm.predict_proba(X_val)[:, 1]
test_pred_lgbm = lgbm.predict_proba(X_test)[:, 1]

print(f"LGBM Val AUC: {roc_auc_score(y_val, val_pred_lgbm):.6f}")



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import lightgbm as lgb

# --- Params (reuse yours) ---
lgbm_params = dict(
    n_estimators=44000,
    class_weight='balanced',
    learning_rate=0.069,
    num_leaves=128,
    max_depth=12,
    min_child_samples=12,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=0.3,
    max_bin=5000,
    random_state=2003,
    boosting_type='gbdt',
    metric='auc',
    verbosity=-1,
    # device='gpu',
)

# --- KFold config ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2003)

# Preallocate
oof_pred = np.zeros(X_train.shape[0], dtype=float)
test_pred = np.zeros(X_test.shape[0], dtype=float)
fold_aucs = []
models = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_trn, y_trn = X_train[trn_idx], y_train[trn_idx]
    X_val_f, y_val_f = X_train[val_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**lgbm_params)

    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val_f, y_val_f)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    # OOF predictions for this fold (use best_iteration_)
    oof_pred[val_idx] = model.predict_proba(
        X_val_f, num_iteration=model.best_iteration_
    )[:, 1]

    # Test predictions (averaged across folds)
    test_pred += model.predict_proba(
        X_test, num_iteration=model.best_iteration_
    )[:, 1] / n_splits

    # Fold AUC
    fold_auc = roc_auc_score(y_val_f, oof_pred[val_idx])
    fold_aucs.append(fold_auc)
    models.append(model)
    print(f"[Fold {fold}] AUC: {fold_auc:.6f} | best_iteration_: {model.best_iteration_}")

# Overall OOF AUC
oof_auc = roc_auc_score(y_train, oof_pred)
print(f"\nOOF AUC: {oof_auc:.6f}")
print("Fold AUCs:", [f"{a:.6f}" for a in fold_aucs])
print(f"Mean AUC: {np.mean(fold_aucs):.6f} Â± {np.std(fold_aucs):.6f}")


# --- ANN model ---



# def build_ann(input_dim: int) -> keras.Model:
#     inputs = layers.Input(shape=(input_dim,))
#     x = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(inputs)
#     x = layers.Dropout(0.25)(x)
#     x = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = layers.Dropout(0.15)(x)
#     x = layers.Dense(64,  activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(x)
#     x = layers.Dropout(0.10)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer=keras.optimizers.Adam(1e-3),
#                   loss="binary_crossentropy",
#                   metrics=[keras.metrics.AUC(name="auc")])
#     return model


from tensorflow.keras import layers, models, callbacks, regularizers
# Build improved ANN
def build_ann(input_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(32, activation="gelu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model




print(X_train.shape[1])


tf.random.set_seed(42)
ann = build_ann(X_train.shape[1])
cbs = [
    callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=8, min_lr=1e-5, verbose=0),
]
ann.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=128, #512
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=cbs
)

val_pred_ann = ann.predict(X_val, verbose=0).ravel()
test_pred_ann = ann.predict(X_test, verbose=0).ravel()

print(f"ANN  Val AUC: {roc_auc_score(y_val, val_pred_ann):.6f}")


import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


# ====== Build model ======
ann = build_ann(X_train.shape[1])

# ====== Callbacks ======
class BestValAccuracy(callbacks.Callback):
    """Track best validation accuracy achieved during training."""
    def __init__(self):
        super().__init__()
        self.best_val_acc = -np.inf
        self.best_epoch = -1
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

best_val_acc_cb = BestValAccuracy()

cbs = [
    callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=10, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=8, min_lr=1e-5, verbose=1
    ),
    best_val_acc_cb,
]

# ====== Train ======
history = ann.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=128,  # try {64, 128, 256, 512}
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=cbs
)

# ====== Predict ======
val_pred_ann = ann.predict(X_val, verbose=0).ravel()
test_pred_ann = ann.predict(X_test, verbose=0).ravel()

# ====== Metrics (AUC + Accuracy @0.5) ======
val_auc = roc_auc_score(y_val, val_pred_ann)
val_acc_05 = accuracy_score(y_val, (val_pred_ann >= 0.5).astype(int))
print(f"ANN Val AUC: {val_auc:.6f}")
print(f"ANN Val ACC @0.50: {val_acc_05:.6f}")

# ====== Report max validation accuracy seen during training ======
print(f"Max val_accuracy during training: {best_val_acc_cb.best_val_acc:.6f} "
      f"(epoch {best_val_acc_cb.best_epoch})")

# ====== (Optional) Tune threshold on val to maximize accuracy ======
# If your end goal is highest accuracy, find the threshold that maximizes it on the validation set.
thresholds = np.linspace(0.0, 1.0, 1001)  # step of 0.001
val_accs = [accuracy_score(y_val, (val_pred_ann >= t).astype(int)) for t in thresholds]
best_idx = int(np.argmax(val_accs))
best_t = float(thresholds[best_idx])
best_val_acc = float(val_accs[best_idx])
print(f"Best threshold on val for ACC: t={best_t:.3f} -> Val ACC={best_val_acc:.6f}")


# --- Find best validation-weighted blend (LGBM vs ANN) ---
best_w, best_auc = 0.5, -1
for w in np.linspace(0, 1, 41):  # step 0.025
    blend = w * val_pred_lgbm + (1 - w) * val_pred_ann
    auc = roc_auc_score(y_val, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"Best blend on val: w_lgbm={best_w:.3f}, w_ann={1-best_w:.3f} | AUC={best_auc:.6f}")


# Cross-validated stacking with robust logs, timings, early stopping, and a fold summary
import sys, time, gc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import lightgbm as lgb  # <-- added for callbacks

# Helpers 
def log(msg):
    print(msg)
    sys.stdout.flush()

# If you have a factory for ANN, prefer this:
# def build_ann(input_dim): ...
# ann_base = build_ann(X_train.shape[1])

# Otherwise snapshot initial ANN weights once and reuse per fold:
ann_init_weights = ann.get_weights()  # comment if you rebuild with build_ann()

# Clone a pristine LGBM per fold to avoid leakage
lgbm_base = clone(lgbm)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta, oof_y = [], []
test_meta = []
fold_summaries = []


log("Starting Cross-Validated Stacking (verbose logs + early stopping)")


for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
    t_fold = time.time()
    log(f"\n===== Fold {fold} =====")
    log(f"Train indices: {len(tr):,} | Val indices: {len(va):,}")

    # Fresh models per fold 
    # If you have a factory: ann = build_ann(X_train.shape[1])
    ann.set_weights(ann_init_weights)     # reuse same init weights for fairness
    lgbm = clone(lgbm_base)

    #  Train ANN: per-epoch AUC, early stopping, LR schedule 
    from tensorflow.keras import callbacks

    ann_cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=10,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=5,
            min_lr=1e-5, verbose=1
        ),
    ]

    log("Training ANN ...")
    t_ann = time.time()
    ann.fit(
        X_train[tr], y_train[tr],
        validation_data=(X_train[va], y_train[va]),
        epochs=200,
        batch_size=1024,
        verbose=1,              # per-epoch logs (loss, auc, etc.)
        callbacks=ann_cbs,
    )
    ann_time = time.time() - t_ann
    log(f"â�±ï¸� ANN finished in {ann_time:.1f}s")

    # Train LGBM: eval logging + early stopping via callbacks (version-safe)
    lgbm.set_params(n_estimators=2000)     # rely on early stopping to cut short
    # Optional: lgbm.set_params(verbosity=1)  # 0=silent, 1=info (may vary by version)

    log("Training LGBM ...")
    t_lgb = time.time()
    lgbm.fit(
        X_train[tr], y_train[tr],
        eval_set=[(X_train[va], y_train[va])],
        eval_metric="auc",
        # verbose=50,  # <-- removed; not supported in your version
        # early_stopping_rounds=100,  # also remove for max compatibility
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=50),   # prints every 50 iters
        ],
    )
    lgb_time = time.time() - t_lgb
    best_iter = getattr(lgbm, "best_iteration_", None)
    log(f" LGBM finished in {lgb_time:.1f}s | best_iteration={best_iter}")

    # Validation predictions
    log("Predicting on validation set ...")
    p_ann = ann.predict(X_train[va], verbose=0).ravel()
    p_lgb = lgbm.predict_proba(X_train[va])[:, 1]
    oof_meta.append(np.c_[p_lgb, p_ann])
    oof_y.append(y_train[va])

    fold_auc_ann = roc_auc_score(y_train[va], p_ann)
    fold_auc_lgb = roc_auc_score(y_train[va], p_lgb)
    log(f"Fold {fold} AUCs â†’ ANN: {fold_auc_ann:.5f} | LGBM: {fold_auc_lgb:.5f}")

    # Test-time meta features per fold 
    log("Predicting on test set ...")
    test_meta.append(np.c_[
        lgbm.predict_proba(X_test)[:, 1],
        ann.predict(X_test, verbose=0).ravel()
    ])

    fold_time = time.time() - t_fold
    fold_summaries.append({
        "fold": fold,
        "ann_auc": float(fold_auc_ann),
        "lgbm_auc": float(fold_auc_lgb),
        "ann_time_s": round(ann_time, 2),
        "lgbm_time_s": round(lgb_time, 2),
        "fold_time_s": round(fold_time, 2),
        "lgbm_best_iter": int(best_iter) if best_iter is not None else None,
    })
    log(f"Fold {fold} done in {fold_time:.1f}s")

    # Free memory between folds (useful for large TF models)
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

# Stack OOF predictions 
oof_meta = np.vstack(oof_meta)
oof_y = np.concatenate(oof_y)
test_meta = np.mean(np.stack(test_meta), axis=0)

log("\nTraining Meta-Model (Logistic Regression) ")
stacker = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
stacker.fit(oof_meta, oof_y)

oof_auc = roc_auc_score(oof_y, stacker.predict_proba(oof_meta)[:, 1])
log(f"\nğŸ�¯ OOF AUC (stacker): {oof_auc:.6f}")

# Final blended predictions
test_blend = stacker.predict_proba(test_meta)[:, 1]
log("âœ… Stacking complete! Test blend predictions ready.")

# Optional: compact fold summary table
log("\n===== Fold Summary =====")
header = f"{'Fold':>4} | {'ANN AUC':>8} | {'LGBM AUC':>8} | {'ANN s':>7} | {'LGBM s':>7} | {'Fold s':>7} | {'BestIter':>8}"
log(header)
log("-"*len(header))
for s in fold_summaries:
    log(f"{s['fold']:>4} | {s['ann_auc']:8.5f} | {s['lgbm_auc']:8.5f} | "
        f"{s['ann_time_s']:7.2f} | {s['lgbm_time_s']:7.2f} | {s['fold_time_s']:7.2f} | "
        f"{str(s['lgbm_best_iter']):>8}")



# Blend test predictions with that weight
test_pred_blend = best_w * test_pred_lgbm + (1 - best_w) * test_pred_ann

# Save Kaggle submission (probabilities)
sub = pd.DataFrame({
    "id": test["id"],
    "y": test_pred_blend
})
# fname = f"submission-ensemble-blend-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
fname = "submission.csv"
sub.to_csv(fname, index=False)
print("Saved:", fname)

# (optional) quick diagnostics file
diag = pd.DataFrame({
    "id": test["id"],
    "pred_lgbm": test_pred_lgbm,
    "pred_ann": test_pred_ann,
    "pred_blend": test_pred_blend
})
diag_fname = f"diagnostics-ensemble-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
diag.to_csv(diag_fname, index=False)
print("Saved:", diag_fname)



# GRAPHS: ROC, PR, Calibration, Weight sweep, Histograms, KS, Importances 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

# ensure we have blended val preds using best_w
val_pred_blend = best_w * val_pred_lgbm + (1 - best_w) * val_pred_ann

# ROC Curves
plt.figure(figsize=(6,5))
for name, prob in [
    ("LGBM",  val_pred_lgbm),
    ("ANN",   val_pred_ann),
    ("Blend", val_pred_blend),
]:
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
plt.plot([0,1], [0,1], ls="--", lw=1, color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (Validation)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Precisionâ€“Recall Curves
plt.figure(figsize=(6,5))
for name, prob in [
    ("LGBM",  val_pred_lgbm),
    ("ANN",   val_pred_ann),
    ("Blend", val_pred_blend),
]:
    precision, recall, _ = precision_recall_curve(y_val, prob)
    ap = average_precision_score(y_val, prob)
    plt.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precisionâ€“Recall curves (Validation)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Calibration (Reliability) Curve 
plt.figure(figsize=(6,5))
for name, prob in [
    ("LGBM",  val_pred_lgbm),
    ("ANN",   val_pred_ann),
    ("Blend", val_pred_blend),
]:
    frac_pos, mean_pred = calibration_curve(y_val, prob, n_bins=15, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=name)
plt.plot([0,1],[0,1], "--", color="gray")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration (Validation)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Weight Sweep: AUC vs LGBM weight 
ws = np.linspace(0, 1, 81)
aucs = [roc_auc_score(y_val, w*val_pred_lgbm + (1-w)*val_pred_ann) for w in ws]
plt.figure(figsize=(6,4))
plt.plot(ws, aucs)
plt.axvline(best_w, ls="--", color="gray")
plt.scatter([best_w], [max(aucs)], zorder=3)
plt.xlabel("Weight on LGBM (w)")
plt.ylabel("Validation AUC")
plt.title("Blend weight sweep")
plt.grid(alpha=0.3)
plt.show()


# Score Histograms (per class)
def plot_score_hist(prob, name):
    plt.figure(figsize=(6,4))
    plt.hist(prob[y_val==0], bins=40, alpha=0.6, label="y=0")
    plt.hist(prob[y_val==1], bins=40, alpha=0.6, label="y=1")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"{name} â€“ score distribution (Validation)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_score_hist(val_pred_lgbm, "LGBM")
plot_score_hist(val_pred_ann,  "ANN")
plot_score_hist(val_pred_blend,"Blend")



# KS Statistic (Validation)
def ks_plot(prob, name):
    # empirical CDFs
    s0 = np.sort(prob[y_val==0])
    s1 = np.sort(prob[y_val==1])
    x  = np.linspace(0, 1, 1000)
    cdf0 = np.searchsorted(s0, x, side="right")/max(1,len(s0))
    cdf1 = np.searchsorted(s1, x, side="right")/max(1,len(s1))
    ks = np.max(np.abs(cdf1 - cdf0))
    plt.figure(figsize=(6,4))
    plt.plot(x, cdf1, label="CDF y=1")
    plt.plot(x, cdf0, label="CDF y=0")
    plt.fill_between(x, cdf1, cdf0, alpha=0.15)
    plt.title(f"{name} â€“ KS={ks:.4f} (Validation)")
    plt.xlabel("Predicted probability")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    # numeric check with scipy
    d, p = ks_2samp(prob[y_val==1], prob[y_val==0])
    print(f"{name} KS-2sample: D={d:.4f}, p={p:.3g}")

ks_plot(val_pred_lgbm, "LGBM")
ks_plot(val_pred_ann,  "ANN")
ks_plot(val_pred_blend,"Blend")




# LGBM Feature Importances (Top 25)
# We trained with numpy arrays, but we *do* know the feature order we built:
#   list(num_features) + list(onehot_feature_names) + binary_cols
# Rebuild names and align to lgbm.feature_importances_
feature_names = list(num_features) + list(encoder.get_feature_names_out(onehot_cols)) + binary_cols
importances = getattr(lgbm, "feature_importances_", None)

if importances is not None and len(importances) == len(feature_names):
    fi = (pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(25))
    plt.figure(figsize=(7,8))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.title("LGBM Feature Importances (Top 25)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Feature importances not available or length mismatch.")


