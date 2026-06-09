import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os



# !pip install --upgrade scikit-learn==1.7.2
# import sklearn
# print(sklearn.__version__)
# !pip install --upgrade lightgbm
print(lgb.__version__)



print("Loading data...")

# Auto-detect Kaggle environment
if os.path.exists("/kaggle/input"):
    TRAIN_TRANS = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENT = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
else:
    TRAIN_TRANS = "train_transaction.csv"
    TRAIN_IDENT = "train_identity.csv"

train_tr = pd.read_csv(TRAIN_TRANS, index_col="TransactionID")
train_id = pd.read_csv(TRAIN_IDENT, index_col="TransactionID")

train = train_tr.merge(train_id, how="left", left_index=True, right_index=True)

# target
y = train["isFraud"]
X = train.drop("isFraud", axis=1)

# ---------------------------
#  Feature list
# ---------------------------
new_features = [
            "TransactionID",
            "TransactionAmt",
            "DeviceInfo",
            "TransactionDT",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
        ]
# Filter only available columns
features = [f for f in new_features if f in X.columns]
X = X[features]

# ---------------------------
# Fill NA
# ---------------------------
X = X.fillna(-999)

# ---------------------------
# Label Encoding
# ---------------------------
label_encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(features, "model_features.pkl")

# ---------------------------
# Train/Val Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# LightGBM Model
# ---------------------------
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "verbosity": -1,
}

model = lgb.train(
    params,
    train_ds,
    valid_sets=[train_ds, val_ds],
    num_boost_round=4000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=300),
        lgb.log_evaluation(period=200)
    ]
)


model.save_model("ieee_lgbm.txt")

# Evaluate Validation AUC
val_pred = model.predict(X_val)
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)
print("MODEL TRAINED AND SAVED SUCCESSFULLY!")



# Auto-detect Kaggle
if os.path.exists("/kaggle/input"):
    TEST_TRANS = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENT = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
else:
    TEST_TRANS = "test_transaction.csv"
    TEST_IDENT = "test_identity.csv"

# Load test data
test_tr = pd.read_csv(TEST_TRANS, index_col="TransactionID")
test_id = pd.read_csv(TEST_IDENT, index_col="TransactionID")
test = test_tr.merge(test_id, how="left", left_index=True, right_index=True)

MODEL_PATH = "/kaggle/working/ieee_lgbm.txt"
FEATURES_PATH = "/kaggle/working/model_features.pkl"
ENCODERS_PATH = "/kaggle/working/label_encoders.pkl"

model = lgb.Booster(model_file=MODEL_PATH)
features = joblib.load(FEATURES_PATH)
label_encoders = joblib.load(ENCODERS_PATH)


# -----------------------------
# Choose only a subset of features for testing
# -----------------------------
selected_features = [
            "TransactionID",
            "TransactionAmt",
            "DeviceInfo",
            "TransactionDT",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
        ]

# Keep only features that exist in test
existing_features = [f for f in selected_features if f in test.columns]
X_test = test[existing_features].copy()

# -----------------------------
# Preprocess categorical features
# -----------------------------
for col in X_test.columns:
    if col in label_encoders:
        le = label_encoders[col]
        # Replace unseen labels with a placeholder
        X_test[col] = X_test[col].astype(str).map(lambda x: x if x in le.classes_ else "unseen_before_label")
        # Refit the encoder to include the placeholder if missing
        if "unseen_before_label" not in le.classes_:
            le_classes = list(le.classes_) + ["unseen_before_label"]
            le.classes_ = np.array(le_classes)
        X_test[col] = le.transform(X_test[col])

# -----------------------------
# Fill missing numeric features with -999
# -----------------------------
for col in X_test.columns:
    if col not in label_encoders:  # numeric
        X_test[col] = X_test[col].fillna(-999)

# -----------------------------
# Make predictions
# -----------------------------
preds = model.predict(X_test)
auc = roc_auc_score(y_val, preds)
# submission = pd.DataFrame({
#     "TransactionID": X_test.index,
#     "isFraud": preds
# })

# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")



\


# =====================================================
#   FULL CODE: LOAD MODEL, LOAD DATA, PREPROCESS, PLOT
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

print("Loading saved model and assets...")

# -----------------------------
# Paths for model + preprocess
# -----------------------------
MODEL_PATH = "/kaggle/input/lggggg/ieee_lgbm.txt"
FEATURES_PATH = "/kaggle/input/lggggg/model_features.pkl"
ENCODERS_PATH = "/kaggle/input/lggggg/label_encoders.pkl"

model = lgb.Booster(model_file=MODEL_PATH)
features = joblib.load(FEATURES_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# -----------------------------
# Load TRAIN dataset again
# (Required to rebuild X_val, y_val)
# -----------------------------
print("Reloading training data...")

if os.path.exists("/kaggle/input"):
    TRAIN_TRANS = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENT = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
else:
    TRAIN_TRANS = "train_transaction.csv"
    TRAIN_IDENT = "train_identity.csv"

train_tr = pd.read_csv(TRAIN_TRANS, index_col="TransactionID")
train_id = pd.read_csv(TRAIN_IDENT, index_col="TransactionID")
train = train_tr.merge(train_id, how="left", left_index=True, right_index=True)

y = train["isFraud"]
X = train[features].copy()

# -----------------------------
# Preprocess exactly like train
# -----------------------------
print("Applying preprocessing...")

for col in X.columns:
    if col in label_encoders:
        le = label_encoders[col]

        X[col] = X[col].astype(str).map(
            lambda x: x if x in le.classes_ else "unseen_before_label"
        )

        if "unseen_before_label" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unseen_before_label")

        X[col] = le.transform(X[col])
    else:
        X[col] = X[col].fillna(-999)

# -----------------------------
# Recreate Train/Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Generating evaluation curves...")

# =====================================================
# 1 & 2 — Training vs Validation AUC & Logloss Curves
# =====================================================
results = model.evals_result_

# ---- AUC CURVE ----
plt.figure(figsize=(10, 5))
plt.plot(results["training"]["auc"], label="Train AUC")
plt.plot(results["valid_1"]["auc"], label="Validation AUC")
plt.xlabel("Iterations")
plt.ylabel("AUC")
plt.title("Training vs Validation AUC Curve")
plt.legend()
plt.grid(True)
plt.show()

# ---- LOGLOSS CURVE ----
plt.figure(figsize=(10, 5))
plt.plot(results["training"]["binary_logloss"], label="Train Logloss")
plt.plot(results["valid_1"]["binary_logloss"], label="Validation Logloss")
plt.xlabel("Iterations")
plt.ylabel("Logloss")
plt.title("Training vs Validation Logloss Curve")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 4 — PRECISION–RECALL CURVE
# =====================================================
val_pred = model.predict(X_val)

precision, recall, thresholds = precision_recall_curve(y_val, val_pred)
ap = average_precision_score(y_val, val_pred)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve (AP = {ap:.4f})")
plt.grid(True)
plt.show()

# =====================================================
# 5 — FEATURE IMPORTANCE
# =====================================================
plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("Top 20 Important Features (Gain Importance)")
plt.show()

print("All curves generated successfully!")



import os
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

print("Loading data...")

# Auto-detect Kaggle environment
if os.path.exists("/kaggle/input"):
    TRAIN_TRANS = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENT = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
else:
    TRAIN_TRANS = "train_transaction.csv"
    TRAIN_IDENT = "train_identity.csv"

train_tr = pd.read_csv(TRAIN_TRANS, index_col="TransactionID")
train_id = pd.read_csv(TRAIN_IDENT, index_col="TransactionID")

train = train_tr.merge(train_id, how="left", left_index=True, right_index=True)

# target
y = train["isFraud"]
X = train.drop("isFraud", axis=1)

# ---------------------------
#  Feature list
# ---------------------------
new_features = [
    "TransactionID", "TransactionAmt", "DeviceInfo", "TransactionDT",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"
]
features = [f for f in new_features if f in X.columns]
X = X[features]

# ---------------------------
# Fill NA
# ---------------------------
X = X.fillna(-999)

# ---------------------------
# Label Encoding
# ---------------------------
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(features, "model_features.pkl")

# ---------------------------
# Train/Val Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# LightGBM Model
# ---------------------------
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val)

evals_result = {}  # dictionary to store evaluation results

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "verbosity": -1,
}

model = lgb.train(
    params,
    train_ds,
    valid_sets=[train_ds, val_ds],
    valid_names=["train", "valid"],
    num_boost_round=3000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=200),
        lgb.record_evaluation(evals_result)  # <-- store metrics here
    ]
)

model.save_model("ieee_lgbm.txt")
joblib.dump(evals_result, "evals_result.pkl")

# ---------------------------
# Evaluate Validation AUC
# ---------------------------
val_pred = model.predict(X_val)
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)

# ============================================================
# 1. TRAINING vs VALIDATION AUC CURVE
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(evals_result["train"]["auc"], label="Train AUC")
plt.plot(evals_result["valid"]["auc"], label="Validation AUC")
plt.title("AUC Curve")
plt.xlabel("Iterations")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.savefig("curve_auc.png", dpi=300)
plt.show()

# ============================================================
# 2. TRAINING vs VALIDATION LOGLOSS CURVE
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(evals_result["train"]["binary_logloss"], label="Train Logloss")
plt.plot(evals_result["valid"]["binary_logloss"], label="Valid Logloss")
plt.title("Logloss Curve")
plt.xlabel("Iterations")
plt.ylabel("Logloss")
plt.legend()
plt.grid(True)
plt.savefig("curve_logloss.png", dpi=300)
plt.show()

# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================
importance = model.feature_importance(importance_type="gain")
indices = np.argsort(importance)[::-1]
feat_names_sorted = X.columns[indices]

plt.figure(figsize=(10, 7))
plt.barh(feat_names_sorted, importance[indices])
plt.title("Feature Importance (Gain)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# ============================================================
# 4. PRECISION–RECALL CURVE
# ============================================================
precision, recall, thresholds = precision_recall_curve(y_val, val_pred)
ap = average_precision_score(y_val, val_pred)

plt.figure(figsize=(10, 5))
plt.plot(recall, precision)
plt.title(f"Precision-Recall Curve (AP = {ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("precision_recall.png", dpi=300)
plt.show()

print("ALL PLOTS SAVED!")
print(" - curve_auc.png")
print(" - curve_logloss.png")
print(" - feature_importance.png")
print(" - precision_recall.png")



import pandas as pd
import os
# Auto-detect Kaggle environment
if os.path.exists("/kaggle/input"):
    TRAIN_TRANS = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENT = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
else:
    TRAIN_TRANS = "train_transaction.csv"
    TRAIN_IDENT = "train_identity.csv"

train_tr = pd.read_csv(TRAIN_TRANS, index_col="TransactionID")
train_id = pd.read_csv(TRAIN_IDENT, index_col="TransactionID")




train = train_tr.merge(train_id, how="left", left_index=True, right_index=True)
features = [
    "TransactionID", "TransactionAmt", "DeviceInfo", "TransactionDT",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"
]
features = [f for f in features if f in train.columns]
train = train[features]
train.head(5)

