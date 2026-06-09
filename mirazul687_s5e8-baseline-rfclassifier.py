# Basic imports
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Encoding, evaluation, model selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Model
import xgboost as xgb

# Misc
import warnings
warnings.filterwarnings("ignore")



# Load competition data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Load original Bank Marketing dataset
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')


print("Train Set Shape:", train.shape)
train


print("Test Set Shape:", test.shape)
test


print("Original Set Shape:", original.shape)
original


sample_submission


print("Unique values in original['y']:", original["y"].unique())
print("Unique values in train['y']:", train["y"].unique())

# Drop 'id' column from train
train = train.drop(columns=["id"])

# Map 'yes'/'no' to 1/0 in original dataset
original["y"] = original["y"].map({"no": 0, "yes": 1})

# Reorder original columns to match train
original = original[train.columns]

# Concatenate both datasets
merged = pd.concat([train, original], ignore_index=True)

# Confirm everything is numeric and merged
print("Merged shape:", merged.shape)
print("Merged target distribution:\n", merged["y"].value_counts())


# Show data types of each column
print(merged.dtypes)

# List of numeric columns
numeric_cols = merged.select_dtypes(include=["int64", "float64"]).columns.tolist()

# List of categorical (object/string) columns
categorical_cols = merged.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)


# Check for missing values in each column
missing_values = merged.isnull().sum()

# Show only columns with at least one missing value
missing_values = missing_values[missing_values > 0]

# Display
print("Missing values:\n", missing_values)



# Identify categorical columns
categorical_cols = merged.select_dtypes(include=["object"]).columns

# Apply Label Encoding to each categorical column
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    merged[col] = le.fit_transform(merged[col])
    label_encoders[col] = le



# Separate features and target
X = merged.drop(columns=["y"])
y = merged["y"]


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

rf_params = {
    'n_estimators': 300,
    'max_depth': 13,
    'min_samples_split': 14,
    'max_features': 0.6106,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(**rf_params)

    # Fit model
    model.fit(X_train, y_train)

    # Predict class and probabilities
    preds_class = model.predict(X_val)
    preds_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    acc = accuracy_score(y_val, preds_class)
    auc = roc_auc_score(y_val, preds_proba)

    print(f"Fold {fold} - Accuracy: {acc:.4f} - ROC AUC: {auc:.4f}")
    fold += 1



# 1. Drop test id
test_id = test["id"]  # Save for submission
test = test.drop(columns=["id"])

# 2. Apply same LabelEncoders to test set
for col in categorical_cols:
    le = label_encoders[col]
    test[col] = le.transform(test[col])  # same categories as train

# 3. Final X and y for training
X_final = merged.drop(columns=["y"])
y_final = merged["y"]

# 4. Train XGBoost model on full data
final_model = xgb.XGBClassifier(**rf_params)

final_model.fit(
    X_final, y_final,
    eval_set=[(X_final, y_final)],
    early_stopping_rounds=50,
    eval_metric="auc",
    verbose=False
)


# 5. Predict probabilities on test set (for class=1)
test_preds_proba = final_model.predict_proba(test)[:, 1]

# 6. Create submission file using raw probabilities
submission = pd.DataFrame({
    "id": test_id,
    "y": test_preds_proba
})

# 7. Save to CSV
submission.to_csv("submission.csv", index=False)
print(" submission.csv file created")



print(submission.head())

