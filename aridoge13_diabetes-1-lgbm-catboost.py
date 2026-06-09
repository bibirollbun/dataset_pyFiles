import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


# Define Categoricals
cat_cols = ["gender",
           "ethnicity",
           "education_level",
           "income_level",
           "smoking_status",
           "employment_status"]


# Handle Missing Values
skip_cols = {"id", "diagnosed_diabetes"}
for col in train.columns:
    if col in skip_cols:
        continue
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Missing")
        if col in test.columns:
            test[col] = test[col].fillna("Missing")
    else:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)


# Encode Categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    # transform test with unseen -> "Unknown"
    test[col] = test[col].astype(str).map(lambda s: s if s in le.classes_ else "Unknown")
    if "Unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "Unknown")
    test[col] = le.transform(test[col])
    encoders[col] = le



# Feature Engineering
print("Features engineered:")

def add_features(df):
    # --- Blood pressure derived ---
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3)

    # --- Lipid ratios (clinically strong predictors) ---
    # Protective ratio
    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    # Atherogenic index
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
    # Triglyceride/HDL ratio – strong metabolic syndrome indicator
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)

    # --- Obesity / metabolic features ---
    # Central adiposity augmentation
    df["bmi_waist_hip"] = df["bmi"] * df["waist_to_hip_ratio"]
    df["bmi_squared"] = df["bmi"] ** 2  # handles non-linear obesity effects

    # --- Lifestyle interactions ---
    df["activity_screen_ratio"] = (
        df["physical_activity_minutes_per_week"] / (df["screen_time_hours_per_day"] + 1e-3)
    )
    df["sleep_screen_product"] = (
        df["sleep_hours_per_day"] * df["screen_time_hours_per_day"]
    )
    df["alcohol_smoking_interaction"] = (
    df["alcohol_consumption_per_week"] * df["smoking_status"]
)

    # --- Cardiometabolic stress score ---
    df["cardio_stress_score"] = (
        df["pulse_pressure"]
        + df["heart_rate"]
        + df["chol_hdl_ratio"]
        + df["tg_hdl_ratio"]
    )

    # --- Socioeconomic bins (ordinal encodings) ---
    # Convert explicitly to ordered integers (if LightGBM can't handle strings directly)
    for col in ["education_level", "income_level", "employment_status"]:
        df[col + "_ord"] = df[col].astype("category").cat.codes

    # --- Family & medical history interactions ---
    df["metabolic_risk_load"] = (
        df["family_history_diabetes"]
        + df["hypertension_history"]
        + df["cardiovascular_history"]
    )

    return df


train = add_features(train)
test  = add_features(test)

# Feature and Target Separation
X = train.drop(columns=["id", "diagnosed_diabetes"])
y = train["diagnosed_diabetes"].astype(int)
X_test = test.drop(columns=["id"], errors="ignore")


# Ensure same column order
X_test = X_test.reindex(columns=X.columns, fill_value=0)
cat_features = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]


# Stratified K folds
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")

    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # LGBM
    model = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(200), log_evaluation(0)]
    )

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=3500,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        bagging_temperature=0.5,
        random_strength=1,
        border_count=128,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        task_type="CPU"
    )

    cat_model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
    early_stopping_rounds=200,
    verbose=100
)

    # Predictions
    lgb_oof = model.predict_proba(X_val)[:, 1]
    lgb_test = model.predict_proba(X_test)[:, 1]

    cat_oof = cat_model.predict_proba(X_val)[:, 1]
    cat_test = cat_model.predict_proba(X_test)[:, 1]

    # Blending
    fold_oof = 0.7 * lgb_oof + 0.3 * cat_oof
    fold_auc = roc_auc_score(y_val, fold_oof)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

    oof_preds[valid_idx] = fold_oof
    test_preds += (0.7 * lgb_test + 0.3 * cat_test) / skf.n_splits



# Ensure predictions are valid probabilities
test_preds = np.clip(test_preds, 0.0, 1.0)
oof_preds  = np.clip(oof_preds, 0.0, 1.0)

# CV summary
cv_auc = roc_auc_score(y, oof_preds)
cv_pr  = average_precision_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")
print(f"Overall CV PR-AUC:  {cv_pr:.4f}")

# Feature Importance
feature_names = X.columns.tolist() 
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=feature_importance.head(30), y="feature", x="importance", orient="h")
plt.title("LightGBM Feature Importance (last fold)")
plt.tight_layout()
plt.show()


# Save Model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/lgbm_last_fold.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
print("Saved model + encoders to artifacts/")


# Create Submission
test_preds = np.clip(test_preds, 0.0, 1.0)
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

