# =========================================
# UFC FIGHT OUTCOME - CATBOOST (FIXED DATE)
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. LOAD DATA
# ===============================
train = pd.read_csv("/kaggle/input/ufc-fight-outcome-prediction-challenge/train_set.csv")
test  = pd.read_csv("/kaggle/input/ufc-fight-outcome-prediction-challenge/test_set.csv")

test_ids = test["Fight_ID"]

# ===============================
# 2. FIX DATE PARSING (KEY FIX)
# ===============================
train["Date"] = pd.to_datetime(train["Date"], format="mixed", errors="coerce")
test["Date"]  = pd.to_datetime(test["Date"],  format="mixed", errors="coerce")

train["Year"] = train["Date"].dt.year
test["Year"]  = test["Date"].dt.year

# ===============================
# 3. LOCATION FEATURE
# ===============================
train["Event_Country"] = train["Location"].str.split(",").str[-1].str.strip()
test["Event_Country"]  = test["Location"].str.split(",").str[-1].str.strip()

# ===============================
# 4. DROP IDENTIFIERS / LEAKAGE
# ===============================
DROP_COLS = [
    "Fight_ID", "Date",
    "Fighter_A_Name", "Fighter_B_Name",
    "Referee", "Location"
]

train = train.drop(columns=DROP_COLS)
test  = test.drop(columns=DROP_COLS)

# ===============================
# 5. DIFFERENCE FEATURES
# ===============================
def build_features(df):
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col.startswith("Fighter_A_"):
            b = col.replace("Fighter_A_", "Fighter_B_")
            if b in df.columns:
                out[col.replace("Fighter_A_", "") + "_DIFF"] = df[col] - df[b]

    # Contextual features
    out["Title_Bout"] = df["Title_Bout"]
    out["Season"] = df["Season"]
    out["Year"] = df["Year"]
    out["Weight_Class"] = df["Weight_Class"]
    out["Event_Country"] = df["Event_Country"]

    return out

X = build_features(train)
y = train["Winner_A"]
X_test = build_features(test)

print("Total features:", X.shape[1])

# ===============================
# 6. CATEGORICAL FEATURES (CATBOOST)
# ===============================
cat_features = [
    X.columns.get_loc("Season"),
    X.columns.get_loc("Weight_Class"),
    X.columns.get_loc("Event_Country")
]

# ===============================
# 7. TRAIN / VALIDATION SPLIT
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 8. CATBOOST MODEL
# ===============================
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100
)

# ===============================
# 9. EVALUATION
# ===============================
preds = model.predict(X_val)
accuracy = accuracy_score(y_val, preds)

print("\n✅ CATBOOST VALIDATION ACCURACY:", round(accuracy, 4))

# ===============================
# 10. TRAIN FULL & SUBMIT
# ===============================
model.fit(X, y, cat_features=cat_features)

test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "Fight_ID": test_ids,
    "Winner_A": test_preds.astype(int)
})

submission.to_csv("ufc_submission_catboost_fixed.csv", index=False)
print("\nSaved: ufc_submission_catboost_fixed.csv")
print(submission.head())


