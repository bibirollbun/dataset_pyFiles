import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- Configuration ---
BASE_DIR    = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR     = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TEST_CSV    = os.path.join(BASE_DIR, "test.csv")
SUBMISSION  = "submission_xgb2.csv"
MODEL_FILE  = "xgb_model.json" # <-- Added model filename

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)


def load_and_flatten(path):
    """
    Load a .npy file (or raw .npy if header corrupt), flatten to 1D,
    pad by repeating last value if too short, or truncate if too long.
    """
    try:
        arr = np.load(path)
        flat = arr.ravel()
    except Exception:
        flat = np.fromfile(path, dtype=np.float32)

    # pad/truncate to TARGET_SIZE
    if flat.size < TARGET_SIZE:
        if flat.size == 0:
            flat = np.zeros(TARGET_SIZE, dtype=np.float32)
        else:
            pad_vals = np.full(TARGET_SIZE - flat.size, flat[-1], dtype=np.float32)
            flat = np.concatenate([flat, pad_vals])
    else:
        flat = flat[:TARGET_SIZE]

    return flat


def extract_features(df):
    """
    For each row in df (with 'id'), load the patch, fix shape, and compute
    mean reflectance for each of the 125 bands.
    Returns an (n_samples, 125) array.
    """
    features = []
    for fn in df['id']:
        path = os.path.join(NPY_DIR, fn)
        flat = load_and_flatten(path)
        # reshape and compute band means
        cube = flat.reshape(IMG_SHAPE)
        band_means = cube.mean(axis=(0, 1))
        features.append(band_means)
    return np.vstack(features)


# --- 1) Load CSVs ---
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# --- 2) Build feature matrices ---
print("Extracting features for training set...")
X = extract_features(train_df)      # shape (n_train, 125)
y = train_df['label'].values

print("Extracting features for test set...")
X_test = extract_features(test_df) # shape (n_test, 125)

# --- 3) Train‐validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# --- 4) XGBoost Regressor setup ---
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.08,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'  # or 'hist' if no GPU
)

# --- 5) Train with early stopping ---
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='mae',
    early_stopping_rounds=20,
    verbose=True
)

# --- 6) Validation performance ---
y_pred_val = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
print(f"Validation MAE: {val_mae:.4f}")

# --- 7) Save the Trained Model ---
# This is the new section to save your model
print(f"Saving model to {MODEL_FILE}...")
xgb_model.save_model(MODEL_FILE)
print(f"✅ Model saved successfully!")


# --- 8) Predict on test set & save submission ---
y_pred_test = xgb_model.predict(X_test)
y_pred_test = np.clip(np.round(y_pred_test), 1, 100).astype(int)

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": y_pred_test
})
submission_df.to_csv(SUBMISSION, index=False)
print(f"✅ Submission saved to {SUBMISSION}")

