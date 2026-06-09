!pip install pytabkit
!pip install --quiet --upgrade 'scikit-learn<1.3'


import os
import io
import torch
import contextlib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pytabkit import TabM_D_Regressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

# Configuration
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


# File paths
TRAIN_FILE = "/kaggle/input/predicting-the-price-of-diamond/train.csv"
TEST_FILE = "/kaggle/input/predicting-the-price-of-diamond/test.csv"
SUB_FILE = "/kaggle/input/predicting-the-price-of-diamond/submission.csv"

# Model parameters
PARAMS = {
    'device': 'cuda',
    'random_state': 100,
    'verbosity': 0,
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 24,
    'num_emb_type': 'pwl',
    'd_embedding': 12,
    'batch_size': 256,
    'lr': 1e-3,
    'n_epochs': 100,
    'dropout': 0.11,
    'd_block': 256,
    'n_blocks': 3,
    'patience': 10,
    'weight_decay': 1e-2,
}
N_SPLITS = 10


# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
submission_df = pd.read_csv(SUB_FILE)

# Combine train & test (drop price from train)
combined = pd.concat([train_df.drop(columns="price"), test_df], ignore_index=True)
combined.head()


# Identify categorical and numeric features
CATS = combined.select_dtypes(include="object").columns.tolist()
FEATURES = [col for col in combined.columns if col != "id"]

# Encode categorical features
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined[CATS] = encoder.fit_transform(combined[CATS])


# Split back
train_processed = combined.iloc[:len(train_df)]
test_processed = combined.iloc[len(train_df):]

X = train_processed[FEATURES].to_numpy()
y = train_df["price"].to_numpy()
X_test = test_processed[FEATURES].to_numpy()


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds_folds = []

print(f"Starting {N_SPLITS}-fold cross-validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"--- Fold {fold}/{N_SPLITS} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train model (suppress internal debug prints from pytabkit)
    model = TabM_D_Regressor(**PARAMS)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)

    # Validation
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    print(f"RÂ² score (val): {r2_score(y_val, val_preds):.5f}")

    # Test predictions
    test_preds_folds.append(model.predict(X_test))


# Overall OOF RÂ²
oof_r2 = r2_score(y, oof_preds)
print("\n-----------------------")
print(f"OOF RÂ² Score: {oof_r2:.5f}")

# Save OOF predictions
pd.DataFrame({
    "id": train_df["id"],
    "price": oof_preds
}).to_csv("oof.csv", index=False)


# Average test predictions across folds
submission_df["price"] = np.mean(test_preds_folds, axis=0)
submission_df.to_csv("submission.csv", index=False)

submission_df.head()

