import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# === 0. Setup & imports ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tqdm import tqdm
import numpy as np



# === 1. Read Kaggle dataset ===
# Replace with your actual file paths
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")



# Optionally drop rows still missing SMILES or key inputs
train = train.dropna(subset=['SMILES'])


# === 3. Vectorize SMILES with character-level n‑gram TF-IDF ===
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=10_000,
    lowercase=False
)

X = vectorizer.fit_transform(train['SMILES'])
print("TF-IDF shape:", X.shape)

X_test_final = vectorizer.transform(test['SMILES'])


train_imputed = train.copy()
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for col in targets:
    print(f"Imputing for {col}...")

    # Only use rows where the target is known
    known_idx = train_imputed[train_imputed[col].notna()].index
    X_known = X[known_idx]
    y_known = train_imputed.loc[known_idx, col].values

    # Rows needing imputation
    missing_idx = train_imputed[train_imputed[col].isna()].index
    X_missing = X[missing_idx]

    if len(missing_idx) == 0:
        continue

    # Fit KNN only on known targets
    knn = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
    knn.fit(X_known)

    # Get 3 nearest known neighbors for each missing sample
    distances, indices = knn.kneighbors(X_missing)

    # Impute as mean of neighbors' target values
    imputed_values = np.array([
        np.mean(y_known[neighbor_ids]) for neighbor_ids in indices
    ])

    # Fill in the missing values
    train_imputed.loc[missing_idx, col] = imputed_values

    # Final fallback in case anything is still NaN
    train_imputed[col] = train_imputed[col].fillna(train_imputed[col].median())



# === 4. Train‑test split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, train_imputed[targets], test_size=0.2, random_state=42
)


# === 5. Train XGBoost model (multi-output via multi:regression) ===
model = xgb.XGBRegressor(
    tree_method='hist',
    max_depth=9,
    n_estimators=750,
    learning_rate=0.05,
    colsample_bytree=0.5,
    random_state=42,
    objective='reg:squarederror'
)


# Fit each target separately (simplest)
models = {}
for col in targets:
    print(f"Training for {col}...")
    model_i = xgb.XGBRegressor(
        tree_method='hist',
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        colsample_bytree=0.5,
        random_state=42,
        objective='reg:squarederror'
    )
    model_i.fit(X_train, y_train[col], 
                eval_set=[(X_val, y_val[col])],
                early_stopping_rounds=20, verbose=False)
    models[col] = model_i


# === 6. Evaluate on validation set ===
preds_val = pd.DataFrame({
    col: models[col].predict(X_val) for col in targets
})
rmse = mean_squared_error(y_val, preds_val, multioutput='raw_values', squared=False)
print("Validation RMSE per target:", dict(zip(targets, rmse)))
print("Overall RMS error:", mean_squared_error(y_val, preds_val, squared=False))



# === 7. Predict on test set → submission.csv ===
submission = test[['id']].copy()  # Adjust if 'ID' column is named differently
for col in targets:
    submission[col] = models[col].predict(X_test_final)

submission.to_csv("submission.csv", index=False)
print("✅ Generated submission.csv")

