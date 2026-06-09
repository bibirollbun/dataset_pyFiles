# ============================================================
# Tabular Playground Series - S5E11 (Loan Payback Prediction)
# Full code: trains model + generates submission.json
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import json
import os

# ============================================================
# STEP 1: VERIFY INPUT FILES
# ============================================================

print("ğŸ“‚ Listing files in /kaggle/input ...")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ============================================================
# STEP 2: LOAD DATA
# ============================================================

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("\nâœ… Data Loaded Successfully")
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nColumns:", list(train.columns))


# ============================================================
# STEP 3: SET TARGET COLUMN (from dataset)
# ============================================================

target = "loan_paid_back"  # âœ… You confirmed this from dataset columns

X = train.drop(columns=[target])
y = train[target]


# ============================================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ============================================================

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

        # Apply same encoding to test data
        test[col] = le.transform(test[col])

print("\nâœ… Categorical columns encoded")


# ============================================================
# STEP 5: TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nâœ… Model trained successfully!")


# ============================================================
# STEP 7: PREDICT ON TEST SET
# ============================================================

preds = model.predict(test)


# ============================================================
# STEP 8: CREATE submission.json (Required by competition)
# ============================================================

submission_json = [
    {"id": int(row_id), "loan_paid_back": int(pred)}
    for row_id, pred in zip(test["id"], preds)
]

output_path = "/kaggle/working/submission.json"
with open(output_path, "w") as f:
    json.dump(submission_json, f, indent=4)

print(f"\nğŸ�‰ submission.json generated at: {output_path}")
print("â�¡ï¸� Go to right panel â†’ Output â†’ Download / Submit to Kaggle")


# ============================================================
# (Optional) Also create submission.csv (for debugging)
# ============================================================

csv_output_path = "/kaggle/working/submission.csv"
sample_submission["loan_paid_back"] = preds
sample_submission.to_csv(csv_output_path, index=False)

print(f"âœ… submission.csv generated at: {csv_output_path}")


