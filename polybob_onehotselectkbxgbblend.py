


# --- Imports ---
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

# --- Load Data ---
def load_data(sample_test=False):
    data_path = "/kaggle/input/isic-2024-challenge/"
    train_filename = data_path + "train-metadata.csv"
    test_filename = data_path + "test-metadata.csv"

    df_train = pd.read_csv(train_filename, low_memory=False)
    X_test = pd.read_csv(test_filename)

    if sample_test:
        X_test = X_test.sample(n=10000, random_state=42)

    X_train = df_train[X_test.columns.to_list()]
    y_train = df_train['target']

    return X_train, y_train, X_test

# --- Batch-safe Prediction ---
def batch_predict(model, X_test, batch_size=50000):
    preds = []
    for i in range(0, X_test.shape[0], batch_size):
        batch = X_test[i:i+batch_size]
        batch_preds = model.predict_proba(batch)[:, 1]
        preds.extend(batch_preds)
    return np.array(preds)

# --- Load and preprocess ---
X_train_raw, y_train, X_test_raw = load_data(sample_test=False)
ids_test = X_test_raw[["isic_id"]].copy()

# --- Identify columns ---
numeric_cols = X_train_raw.select_dtypes(include='number').columns.tolist()
cat_cols = X_train_raw.select_dtypes(include='object').columns.tolist()

# --- SelectKBest on numeric only ---
X_train_num = X_train_raw[numeric_cols].fillna(X_train_raw[numeric_cols].median())
X_test_num = X_test_raw[numeric_cols].fillna(X_train_raw[numeric_cols].median())

# Scale numeric
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Feature selection
k = min(30, X_train_num_scaled.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_num_selected = selector.fit_transform(X_train_num_scaled, y_train)
X_test_num_selected = selector.transform(X_test_num_scaled)

print(f"✅ Selected {X_train_num_selected.shape[1]} numeric features")

# --- OneHot encode categoricals (sparse) ---
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_cat_encoded = encoder.fit_transform(X_train_raw[cat_cols])
X_test_cat_encoded = encoder.transform(X_test_raw[cat_cols])

# --- Combine selected numeric + full categorical features ---
X_train_all = sparse.hstack([X_train_num_selected, X_train_cat_encoded]).tocsr()
X_test_all = sparse.hstack([X_test_num_selected, X_test_cat_encoded]).tocsr()
print(f"✅ Final shape after combining: {X_train_all.shape}")

# --- K-Fold Ensemble with XGBoost ---
fold_preds = np.zeros(X_test_all.shape[0])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train)):
    X_tr, y_tr = X_train_all[train_idx], y_train.iloc[train_idx]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum()
    )

    model.fit(X_tr, y_tr)
    print(f"✅ Trained fold {fold + 1}")

    fold_preds += batch_predict(model, X_test_all) / 5

# --- Submission ---
submission = pd.DataFrame({
    "isic_id": ids_test["isic_id"],
    "target": np.nan_to_num(fold_preds, nan=0.0)
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")


