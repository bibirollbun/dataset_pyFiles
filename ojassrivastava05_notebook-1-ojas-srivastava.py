# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


df.shape


df.describe


df.columns



df.head(10)


df.describe()


df.isna().sum()


df.columns



sns.boxplot(
    x='diagnosed_diabetes',
    y='alcohol_consumption_per_week',
    data=df
)
df.groupby('diagnosed_diabetes')['alcohol_consumption_per_week'].mean()



sns.violinplot(
    x='diagnosed_diabetes',
    y='alcohol_consumption_per_week',
    data=df
)



df.groupby('diagnosed_diabetes')['alcohol_consumption_per_week'].mean()


df.dtypes


numeric_cols = df.select_dtypes(include=['int64','float64']).columns
numeric_cols = numeric_cols.drop('diagnosed_diabetes')   # exclude target

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='diagnosed_diabetes', y=col, data=df)
    plt.title(f'Boxplot: {col} vs diagnosed_diabetes')
    plt.show()



num_df = df.select_dtypes(include=['int64','float64'])

means = num_df.groupby(df['diagnosed_diabetes']).mean().T
means['difference'] = means[1.0] - means[0.0]
means.sort_values('difference', ascending=False)



from sklearn.model_selection import train_test_split

features = ['triglycerides','ldl_cholesterol','age','cholesterol_total',
            'systolic_bp','bmi','physical_activity_minutes_per_week',
            'hdl_cholesterol','diet_score','family_history_diabetes']

X = df[features]
y = df['diagnosed_diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)



from sklearn.metrics import accuracy_score, roc_auc_score

pred = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)[:,1]

print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))



coef = pd.DataFrame({
    'feature': features,
    'coef': model.coef_[0]
}).sort_values('coef', ascending=False)

print(coef)



# ============================
# Add categorical features pipeline + Logistic Regression baseline
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                             auc, confusion_matrix, classification_report)

# --- 1) Feature selection ---
numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','bmi','physical_activity_minutes_per_week',
    'hdl_cholesterol','diet_score'
    # Note: family_history_diabetes is binary; we will include it as numeric or categorical below
]

# Choose categorical features to add
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status'
]

# Include the important binary feature as numeric
binary_features = ['family_history_diabetes']

# Final X, y
features = numeric_features + binary_features + categorical_features
X = df[features].copy()          # assuming `df` is your dataframe in the notebook
y = df['diagnosed_diabetes']

# --- 2) Train-test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --- 3) Build preprocessing pipelines ---
# Numeric pipeline: median impute -> scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: most frequent impute -> one-hot encode (drop first to avoid collinearity)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features + binary_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='drop')   # drop any other columns

# --- 4) Full pipeline with classifier ---
pipe = Pipeline([
    ('preproc', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', class_weight=None))
])

# (Optional) if your classes are imbalanced consider: class_weight='balanced' inside LogisticRegression
# e.g. LogisticRegression(max_iter=1000, class_weight='balanced')

# --- 5) Fit ---
pipe.fit(X_train, y_train)

# --- 6) Evaluate ---
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# --- 7) Feature names and coefficients (map one-hot names back to columns) ---
# This requires sklearn >= 1.0 to use get_feature_names_out on ColumnTransformer
try:
    # Get transformed feature names
    ohe = pipe.named_steps['preproc'].named_transformers_['cat'].named_steps['ohe']
    cat_names = ohe.get_feature_names_out(categorical_features)
    num_names = numeric_features + binary_features
    feature_names = np.concatenate([num_names, cat_names])
except Exception:
    # Fallback (older sklearn): build minimal names (not as pretty)
    feature_names = (numeric_features + binary_features) + [f"cat_{i}" for i in range(len(categorical_features))]

# Get coefficients
coefs = pipe.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)

print("\nTop coefficients (absolute value):")
print(coef_df.head(20))

# Save the pipeline for later
import joblib
joblib.dump(pipe, "logreg_cat_pipeline.joblib")
print("\nSaved pipeline to logreg_cat_pipeline.joblib")



test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")   # path may vary



numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','bmi','physical_activity_minutes_per_week',
    'hdl_cholesterol','diet_score'
]

binary_features = ['family_history_diabetes']

categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]

features = numeric_features + binary_features + categorical_features



X_test_final = test_df[features].copy()

# Predict probability of class 1 (diabetes)
test_pred_proba = pipe.predict_proba(X_test_final)[:, 1]



submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_pred_proba
})

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv successfully!")



# XGBoost pipeline: preprocessing (ColumnTransformer) + training with early stopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1) Feature lists (match your training)
# -------------------------
numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','bmi','physical_activity_minutes_per_week',
    'hdl_cholesterol','diet_score'
]
binary_features = ['family_history_diabetes']
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]

features = numeric_features + binary_features + categorical_features

# -------------------------
# 2) Prepare data & splits
# -------------------------
# assume `df` is your training DataFrame already loaded in notebook
X = df[features].copy()
y = df['diagnosed_diabetes']

# initial train/test split (holdout for final eval)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# from train, create train/validation for early stopping (small val set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
)
# This results roughly in: train ~70%, val ~10%, test ~20%

# -------------------------
# 3) Preprocessing (ColumnTransformer)
# -------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # scaler is optional for trees — kept for consistency if you switch models
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))  # drop first to reduce dims
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features + binary_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='drop')

# Fit preprocessor on training data, transform train/val/test
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# If you want to inspect feature names (for importance mapping)
# Works if sklearn >=1.0
try:
    # numeric names
    num_names = numeric_features + binary_features
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_names = ohe.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_names, cat_names])
except Exception:
    # fallback: numeric names then placeholder cat names
    num_names = numeric_features + binary_features
    cat_names = [f"cat_{i}" for i in range(preprocessor.transform(X_train[:1]).shape[1] - len(num_names))]
    feature_names = np.concatenate([num_names, cat_names])

# -------------------------
# 4) Handle class imbalance
# -------------------------
# Compute scale_pos_weight = (neg / pos) to give XGBoost approx balanced objective
n_pos = y_train.sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / max(1, n_pos)
print("scale_pos_weight (train):", scale_pos_weight)

# -------------------------
# 5) XGBoost model (with sensible defaults)
# -------------------------
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    verbosity=1
)

# -------------------------
# 6) Train with early stopping (use validation set)
# -------------------------
xgb.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# -------------------------
# 7) Evaluate on test set
# -------------------------
y_pred_proba = xgb.predict_proba(X_test_proc)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

# -------------------------
# 8) Feature importance (table + bar plot)
# -------------------------
# Use feature_names computed earlier for readable table
importances = xgb.feature_importances_
fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop features by XGBoost importance:")
print(fi.head(30))

# Quick bar plot
plt.figure(figsize=(8,6))
plt.barh(fi['feature'].head(20)[::-1], fi['importance'].head(20)[::-1])
plt.title("Top 20 XGBoost feature importances")
plt.tight_layout()
plt.show()

# -------------------------
# 9) Save preprocessor + model for later inference (pipeline-like)
# -------------------------
joblib.dump({'preprocessor': preprocessor, 'model': xgb}, "xgb_preproc_model.joblib")
print("Saved preprocessor+model to xgb_preproc_model.joblib")

# -------------------------
# 10) Create submission.csv using test.csv
# -------------------------
# Load your Kaggle test set (adjust path if needed)
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# ensure features exist
missing = set(features) - set(test_df.columns)
if missing:
    raise ValueError(f"Missing columns in test.csv: {missing}")

X_test_kaggle = test_df[features].copy()
X_test_kaggle_proc = preprocessor.transform(X_test_kaggle)   # transform using fitted preprocessor
kaggle_preds = xgb.predict_proba(X_test_kaggle_proc)[:, 1]

submission = pd.DataFrame({
    "id": test_df['id'],
    "diagnosed_diabetes": kaggle_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")



# Quick tuned XGBoost pipeline (fast improvement)
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Feature lists (same as before)
numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','bmi','physical_activity_minutes_per_week',
    'hdl_cholesterol','diet_score'
]
binary_features = ['family_history_diabetes']
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]
features = numeric_features + binary_features + categorical_features

# Prepare data (assumes `df` loaded)
X = df[features].copy()
y = df['diagnosed_diabetes']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocessing (identical to earlier pipeline)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())   # optional for trees, but ok
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features + binary_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='drop')

# Fit preprocessor
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# compute scale_pos_weight
n_pos = y_train.sum()
n_neg = (y_train==0).sum()
scale_pos_weight = n_neg / max(1, n_pos)
print("scale_pos_weight:", scale_pos_weight)

# Quick tuned params
xgb_quick = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.7,
    gamma=1.0,
    reg_alpha=3.0,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# Train with early stopping using built-in method
xgb_quick.fit(
    X_train_proc, y_train,
    eval_set=[(X_test_proc, y_test)],
    early_stopping_rounds=50,
    verbose=50
)

# Evaluate
proba = xgb_quick.predict_proba(X_test_proc)[:,1]
pred = (proba >= 0.5).astype(int)
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=4))

# Save preprocessor + model together for inference
joblib.dump({'preprocessor': preprocessor, 'model': xgb_quick}, "xgb_quick_preproc_model.joblib")
print("Saved xgb_quick_preproc_model.joblib")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
X_test_kaggle = test_df[features].copy()
X_test_kaggle_proc = preprocessor.transform(X_test_kaggle)
preds = xgb_quick.predict_proba(X_test_kaggle_proc)[:,1]
submission = pd.DataFrame({"id": test_df['id'], "diagnosed_diabetes": preds})
submission.to_csv("submission_quick_xgb.csv", index=False)
print("Saved submission_quick_xgb.csv")



# === Final training + submission creation ===
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------
# 0) CONFIG - update if needed
# ---------------------------
# If you have Optuna study in the session use: study.best_trial.params
# otherwise paste the best params dict here (example below)
try:
    # If you have 'study' object in session (recommended), this picks best params automatically
    best_params = study.best_trial.params
    print("Loaded best params from `study` object.")
except Exception:
    # Fallback: use a strong tuned set (change if you want)
    best_params = {
        "n_estimators": 10000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "gamma": 1.0,
        "reg_alpha": 3.0,
        "reg_lambda": 2.0
    }
    print("Using fallback tuned params (no study found).")

# ---------------------------
# 1) Feature lists (must match training)
# ---------------------------
numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','bmi','physical_activity_minutes_per_week',
    'hdl_cholesterol','diet_score'
]
binary_features = ['family_history_diabetes']
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]
features = numeric_features + binary_features + categorical_features

# ---------------------------
# 2) GPU detection and device config
# ---------------------------
gpu_available = False
try:
    gpu_available = (os.system("nvidia-smi -L > /dev/null 2>&1") == 0)
except Exception:
    gpu_available = False

if gpu_available:
    best_params["tree_method"] = "hist"
    best_params["device"] = "cuda"   # XGBoost >=2.0 style
else:
    best_params["tree_method"] = "hist"
    best_params["device"] = "cpu"

# Ensure these required keys are set
best_params.setdefault("use_label_encoder", False)
best_params.setdefault("eval_metric", "auc")
best_params.setdefault("random_state", 42)
best_params.setdefault("n_jobs", -1)

print("Final training device:", best_params.get("device"))

# ---------------------------
# 3) Load full training data (assumes df is present)
# ---------------------------
# df must be defined in the notebook (your training DataFrame)
X_full = df[features].copy()
y_full = df['diagnosed_diabetes'].copy()

# ---------------------------
# 4) Preprocessor - fit on full training set
# ---------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features + binary_features),
    ('cat', cat_pipeline, categorical_features)
], remainder='drop')

preprocessor_full = preprocessor.fit(X_full)   # fit on entire training data
X_full_proc = preprocessor_full.transform(X_full)

# ---------------------------
# 5) Compute scale_pos_weight on full data and add to params
# ---------------------------
n_pos_full = int(y_full.sum())
n_neg_full = int((y_full == 0).sum())
scale_pos_weight_full = float(n_neg_full) / max(1.0, float(n_pos_full))
best_params["scale_pos_weight"] = scale_pos_weight_full
print("scale_pos_weight (full):", best_params["scale_pos_weight"])

# ---------------------------
# 6) Train final model with a small train/val split for early stopping
# ---------------------------
# Use a small validation holdout for early stopping (10% of full)
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_full_proc, y_full, test_size=0.10, stratify=y_full, random_state=42
)

# Instantiate model with best params
model = XGBClassifier(**best_params)

# Fit with early stopping on the small validation set
model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    early_stopping_rounds=50,
    verbose=50
)

# Optionally evaluate on the holdout validation
val_proba = model.predict_proba(X_val_final)[:,1]
try:
    val_auc = roc_auc_score(y_val_final, val_proba)
    print("Validation ROC-AUC (final model):", val_auc)
except Exception:
    pass

# ---------------------------
# 7) Save final preprocessor + model
# ---------------------------
joblib.dump({'preprocessor': preprocessor_full, 'model': model}, "xgb_final_preproc_model.joblib")
print("Saved xgb_final_preproc_model.joblib")

# ---------------------------
# 8) Create submission.csv using test.csv
# ---------------------------
test_path = "/kaggle/input/playground-series-s5e12/test.csv"   # change if your test file path differs
if not os.path.exists(test_path):
    raise FileNotFoundError(f"{test_path} not found in working directory. Place Kaggle test CSV here or update path.")

test_df = pd.read_csv(test_path)

# Ensure required features exist in test
missing = set(features) - set(test_df.columns)
if missing:
    raise ValueError(f"Missing columns in test.csv required for prediction: {missing}")

X_test_kaggle = test_df[features].copy()
X_test_kaggle_proc = preprocessor_full.transform(X_test_kaggle)
kaggle_preds = model.predict_proba(X_test_kaggle_proc)[:, 1]

submission = pd.DataFrame({
    "id": test_df['id'],
    "diagnosed_diabetes": kaggle_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (first 5 rows):")
print(submission.head())



import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ------------------------
# 1. Add engineered features to df
# ------------------------
df_eng = df.copy()

df_eng["ldl_hdl_ratio"] = df_eng["ldl_cholesterol"] / (df_eng["hdl_cholesterol"] + 1)
df_eng["chol_trig_ratio"] = df_eng["cholesterol_total"] / (df_eng["triglycerides"] + 1)
df_eng["activity_bmi_ratio"] = df_eng["physical_activity_minutes_per_week"] / (df_eng["bmi"] + 1)
df_eng["waist_bmi_ratio"] = df_eng["waist_to_hip_ratio"] / (df_eng["bmi"] + 1)

df_eng["bmi_age_interaction"] = df_eng["bmi"] * df_eng["age"]
df_eng["diet_activity_interaction"] = df_eng["diet_score"] * df_eng["physical_activity_minutes_per_week"]

df_eng["sleep_screen_ratio"] = df_eng["sleep_hours_per_day"] / (df_eng["screen_time_hours_per_day"] + 1)

df_eng["bp_pulse_pressure"] = df_eng["systolic_bp"] - df_eng["diastolic_bp"]
df_eng["bp_mean_arterial"] = (2 * df_eng["diastolic_bp"] + df_eng["systolic_bp"]) / 3

# ------------------------
# 2. Define full feature list
# ------------------------

numeric_features = [
    'triglycerides','ldl_cholesterol','age','cholesterol_total',
    'systolic_bp','diastolic_bp','bmi','waist_to_hip_ratio',
    'physical_activity_minutes_per_week','sleep_hours_per_day',
    'screen_time_hours_per_day','heart_rate','diet_score',

    # engineered numeric features:
    "ldl_hdl_ratio","chol_trig_ratio","activity_bmi_ratio","waist_bmi_ratio",
    "bmi_age_interaction","diet_activity_interaction",
    "sleep_screen_ratio","bp_pulse_pressure","bp_mean_arterial"
]

binary_features = ['family_history_diabetes']
categorical_features = [
    'gender', 'ethnicity', 'education_level', 'income_level',
    'smoking_status', 'employment_status'
]

features = numeric_features + binary_features + categorical_features

X = df_eng[features]
y = df_eng['diagnosed_diabetes']

# ------------------------
# 3. Preprocessor (same as before but handles more numeric columns)
# ------------------------

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features + binary_features),
    ('cat', cat_pipeline, categorical_features)
])

# ------------------------
# 4. Train/val split for early stopping
# ------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# Fit preprocessor on train only
preprocessor_full = preprocessor.fit(X_train)
X_train_proc = preprocessor_full.transform(X_train)
X_val_proc   = preprocessor_full.transform(X_val)

# ------------------------
# 5. Use Optuna best params (safe mode)
# ------------------------

try:
    best_params = study.best_trial.params
    print("Using Optuna best params.")
except:
    best_params = {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "gamma": 1.0,
        "reg_alpha": 3.0,
        "reg_lambda": 2.0
    }
    print("Using fallback tuned params.")

# GPU detection
gpu_available = False
try:
    gpu_available = (os.system("nvidia-smi -L > /dev/null 2>&1") == 0)
except:
    pass

if gpu_available:
    best_params["tree_method"] = "hist"
    best_params["device"] = "cuda"
else:
    best_params["tree_method"] = "hist"
    best_params["device"] = "cpu"

# add required params
best_params["use_label_encoder"] = False
best_params["eval_metric"] = "auc"
best_params["random_state"] = 42
best_params["n_jobs"] = -1

# scale_pos_weight
n_pos = y_train.sum()
n_neg = (y_train == 0).sum()
best_params["scale_pos_weight"] = float(n_neg) / max(1.0, float(n_pos))

print("scale_pos_weight:", best_params["scale_pos_weight"])

model = XGBClassifier(**best_params)

# ------------------------
# 6. Train with early stopping
# ------------------------

model.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# ------------------------
# 7. Evaluate on validation
# ------------------------
val_proba = model.predict_proba(X_val_proc)[:, 1]
val_auc = roc_auc_score(y_val, val_proba)
print("Validation AUC with engineered features:", val_auc)

# ------------------------
# 8. Save model
# ------------------------
joblib.dump({"preprocessor": preprocessor_full, "model": model}, "xgb_feature_engineered.joblib")
print("Saved xgb_feature_engineered.joblib")

# ------------------------
# 9. Build submission.csv
# ------------------------

test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test_eng = test_df.copy()

# add same engineered features
test_eng["ldl_hdl_ratio"] = test_eng["ldl_cholesterol"] / (test_eng["hdl_cholesterol"] + 1)
test_eng["chol_trig_ratio"] = test_eng["cholesterol_total"] / (test_eng["triglycerides"] + 1)
test_eng["activity_bmi_ratio"] = test_eng["physical_activity_minutes_per_week"] / (test_eng["bmi"] + 1)
test_eng["waist_bmi_ratio"] = test_eng["waist_to_hip_ratio"] / (test_eng["bmi"] + 1)

test_eng["bmi_age_interaction"] = test_eng["bmi"] * test_eng["age"]
test_eng["diet_activity_interaction"] = test_eng["diet_score"] * test_eng["physical_activity_minutes_per_week"]

test_eng["sleep_screen_ratio"] = test_eng["sleep_hours_per_day"] / (test_eng["screen_time_hours_per_day"] + 1)

test_eng["bp_pulse_pressure"] = test_eng["systolic_bp"] - test_eng["diastolic_bp"]
test_eng["bp_mean_arterial"] = (2 * test_eng["diastolic_bp"] + test_eng["systolic_bp"]) / 3

# ensure all features present
X_test_final = preprocessor_full.transform(test_eng[features])
test_preds = model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    "id": test_eng["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(submission.head())





