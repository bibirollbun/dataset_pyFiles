import os
from pathlib import Path

def find_dataset_dir(required=("train.csv", "test.csv")) -> Path:
    """Locate the dataset directory in Kaggle or locally, with helpful diagnostics."""
    # 0) Env var overrides
    env_dir = os.getenv("DATASET_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and all((p/f).exists() for f in required):
            print(f"ğŸ”§ Using DATASET_DIR override: {p}")
            return p
        else:
            print(f"âš ï¸� DATASET_DIR set but invalid: {p}")

    env_name = os.getenv("DATASET_NAME")
    kaggle_input = Path("/kaggle/input")
    if env_name and kaggle_input.exists():
        cand = kaggle_input / env_name
        if cand.is_dir() and all((cand/f).exists() for f in required):
            print(f"ğŸ”§ Using DATASET_NAME under /kaggle/input: {cand}")
            return cand
        else:
            print(f"âš ï¸� DATASET_NAME set but not found or missing files: {cand}")

    candidates = []

    # 1) Exact files at /kaggle/input (rare, but check)
    if kaggle_input.exists():
        if all((kaggle_input/f).exists() for f in required):
            candidates.append(kaggle_input)

        # 2) Any subdir of /kaggle/input containing required files (non-recursive, then recursive)
        for d in kaggle_input.iterdir():
            if d.is_dir() and all((d/f).exists() for f in required):
                candidates.append(d)

        # 3) Recursive search as a last resort (handles nested or renamed datasets)
        if not candidates:
            trains = list(kaggle_input.rglob(required[0]))
            tests  = set(p.parent for p in kaggle_input.rglob(required[1]))
            for t in trains:
                if t.parent in tests:
                    candidates.append(t.parent)

    # 4) Local working directory fallback
    cwd = Path(".").resolve()
    if all((cwd/f).exists() for f in required):
        candidates.append(cwd)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            unique.append(c)

    if not unique:
        # Helpful diagnostics
        print("â�Œ Could not find a folder containing both train.csv and test.csv.")
        if kaggle_input.exists():
            print("ğŸ“� Top-level entries in /kaggle/input:")
            for d in kaggle_input.iterdir():
                print("  -", d.name)
        print("ğŸ”� Searched:", kaggle_input if kaggle_input.exists() else "(no /kaggle/input)")
        print("ğŸ”� Also checked working dir:", cwd)
        raise FileNotFoundError("Could not find dataset directory containing train.csv and test.csv")

    # Pick the most recently modified candidate
    chosen = max(unique, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    # Print a short diagnostic list
    print("âœ… Candidates that contain the required files:")
    for c in unique:
        marker = " (chosen)" if c == chosen else ""
        print(f"  - {c}{marker}")
    return chosen

# Usage:
dataset_dir = find_dataset_dir()
print(f"ğŸ“‚ Using dataset directory: {dataset_dir}")

train_path = dataset_dir / "train.csv"
test_path  = dataset_dir / "test.csv"



# --- Model train + submit (drop in after dataset_dir/train_path/test_path) ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1) Load
df      = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 2) Infer columns by position
id_col     = df.columns[0]     # leftmost
target_col = df.columns[-1]    # rightmost
print(f"ğŸªª ID column: {id_col}")
print(f"ğŸ�¯ Target column: {target_col}")

# 3) Split features/target
X = df.iloc[:, 1:-1]   # everything between ID and target
y = df.iloc[:, -1]

# 4) Preprocess (numeric + categorical)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ],
    remainder="drop"
)

# 5) Model
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# 6) Train/validate
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"âœ… RMSE: {rmse:.4f}")


import matplotlib.pyplot as plt

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in XGBoost Model")
plt.gca().invert_yaxis()
plt.show()



# 7) Predict test + save submission
X_sub = df_test.iloc[:, 1:]  # skip ID
pred = pipe.predict(X_sub)

submission = pd.DataFrame({
    id_col: df_test[id_col],
    target_col: pred
})
print("\nğŸ“„ Submission preview:")
print(submission.head())

submission.to_csv("submission.csv", index=False)
print("\nğŸ’¾ Saved to submission.csv")


