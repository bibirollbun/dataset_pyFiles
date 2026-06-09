# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning models and preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
# ML algorithms (baseline + advanced)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna

# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# 2.1 Load the Dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

# 2.2 Check Shape
print("Shape of training data:", train_df.shape)

# 2.3 Preiew Data
train_df.head()

# 2.4 Column Info
train_df.info()

# 2.5 Missing Values
train_df.isnull().sum()


# 3.1 Summary Statistics
train_df.describe().T

# 3.2 Distribution of Numerical Features
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# 3.3 Categorical Feature Analysis
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

plt.figure(figsize=(15, 16))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.4 Target Variable Balance
print(train_df['y'].value_counts(normalize=True))

plt.figure(figsize=(5,5))
sns.countplot(data=train_df, x='y')
plt.title("Target Variable Distribution (y)")
plt.show()

# 3.5 Correlation Heatmap
plt.figure(figsize=(12,8))
corr = train_df.select_dtypes(include=['int64','float64']).corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


# 4.1 Separate Features and Target
X = train_df.drop(columns=['id','y'])   # drop id (not useful) and target
y = train_df['y']

# 4.2 Encode Categorical Variables
cat_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# 4.3 Scale Numerical Features
num_cols = X.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 4.4 Train-Test Split (no SMOTE here)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_valid.shape)
print("\nTarget distribution in train set:\n", y_train.value_counts(normalize=True))


# 5.1 Logistic Regression
log_reg = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict_proba(X_valid)[:,1]
auc_log = roc_auc_score(y_valid, y_pred_log)

# 5.2 Random Forest
rf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_valid)[:,1]
auc_rf = roc_auc_score(y_valid, y_pred_rf)

# 5.3 XGBoost
xgb = XGBClassifier(
    scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict_proba(X_valid)[:,1]
auc_xgb = roc_auc_score(y_valid, y_pred_xgb)

# 5.4 LightGBM  (rename from `lgb` -> `lgb_base`)
lgb_base = LGBMClassifier(class_weight="balanced", random_state=42, n_estimators=200)
lgb_base.fit(X_train, y_train)
y_pred_lgb = lgb_base.predict_proba(X_valid)[:,1]
auc_lgb = roc_auc_score(y_valid, y_pred_lgb)

# 5.5 Compare Results
print("ROC AUC Scores:")
print(f"Logistic Regression: {auc_log:.4f}")
print(f"Random Forest: {auc_rf:.4f}")
print(f"XGBoost: {auc_xgb:.4f}")
print(f"LightGBM: {auc_lgb:.4f}")


# --- prevent shadowing of the lightgbm module ---
if "lgb" in globals() and not hasattr(lgb, "__package__"):
    # a model instance named `lgb` exists; remove it
    del lgb

import lightgbm as lgb
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
TUNE_SIZE = min(len(X_train), 200_000)   # subsample size for tuning
TEST_SIZE_TUNE = 0.2                     # split inside the tuning sample
N_TRIALS_LGB = 12                        # keep small for speed
N_TRIALS_XGB = 12

# ---- 6.1 Stratified subsample for tuning ----
if len(X_train) > TUNE_SIZE:
    tune_idx, _ = next(iter(
        StratifiedKFold(n_splits=int(len(X_train)/TUNE_SIZE), shuffle=True,
                        random_state=RANDOM_STATE).split(X_train, y_train)
    ))
    X_tune_full = X_train.iloc[tune_idx].copy()
    y_tune_full = y_train.iloc[tune_idx].copy()
else:
    X_tune_full, y_tune_full = X_train.copy(), y_train.copy()

X_tune_tr, X_tune_va, y_tune_tr, y_tune_va = train_test_split(
    X_tune_full, y_tune_full, test_size=TEST_SIZE_TUNE,
    stratify=y_tune_full, random_state=RANDOM_STATE
)

# Imbalance ratio for XGB
pos = y_tune_tr.sum()
neg = len(y_tune_tr) - pos
SCALE_POS_WEIGHT = neg / max(pos, 1)

# ---- 6.2 Fast LGBM tuning (hold-out) ----
def objective_lgb(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tune_tr, y_tune_tr,
        eval_set=[(X_tune_va, y_tune_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
    )
    pred = model.predict_proba(X_tune_va)[:, 1]
    return roc_auc_score(y_tune_va, pred)

study_lgb = optuna.create_study(direction="maximize", study_name="LGBM-fast",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=4))
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS_LGB, show_progress_bar=False)
print("LGBM best AUC (tune hold-out):", study_lgb.best_value)
print("LGBM best params:", study_lgb.best_params)

best_lgb = LGBMClassifier(**study_lgb.best_params, objective="binary",
                          metric="auc", class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1)

best_lgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)]
)
lgb_valid_auc = roc_auc_score(y_valid, best_lgb.predict_proba(X_valid)[:, 1])
print(f"Validation AUC (LGBM tuned fast): {lgb_valid_auc:.5f}")

# ---- 6.3 Fast XGB tuning (hold-out) ----
def objective_xgb(trial):
    params = {
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "scale_pos_weight": float(SCALE_POS_WEIGHT),
        "n_estimators": trial.suggest_int("n_estimators", 250, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    model.fit(
        X_tune_tr, y_tune_tr,
        eval_set=[(X_tune_va, y_tune_va)],
        verbose=False,
        early_stopping_rounds=80
    )
    pred = model.predict_proba(X_tune_va)[:, 1]
    return roc_auc_score(y_tune_va, pred)

study_xgb = optuna.create_study(direction="maximize", study_name="XGB-fast",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=4))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, show_progress_bar=False)
print("XGB best AUC (tune hold-out):", study_xgb.best_value)
print("XGB best params:", study_xgb.best_params)

best_xgb = XGBClassifier(**study_xgb.best_params, tree_method="hist",
                         eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1,
                         scale_pos_weight=float(SCALE_POS_WEIGHT))
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
    early_stopping_rounds=120
)
xgb_valid_auc = roc_auc_score(y_valid, best_xgb.predict_proba(X_valid)[:, 1])
print(f"Validation AUC (XGB tuned fast): {xgb_valid_auc:.5f}")

# ---- 6.4 Pick model for inference ----
print("\n=== Fast Tuned Model Comparison (Validation AUC) ===")
print(f"LightGBM: {lgb_valid_auc:.5f} | XGBoost: {xgb_valid_auc:.5f}")
best_model_name = "LightGBM" if lgb_valid_auc >= xgb_valid_auc else "XGBoost"
print("Selected for inference:", best_model_name)

final_model = best_lgb if best_model_name == "LightGBM" else best_xgb



# 7.1 Load Test Set
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_ids = test_df['id']

# Encode categorical features (same encoders as training)
for col in cat_cols:
    test_df[col] = le.fit_transform(test_df[col])

# Scale numerical columns
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Drop id
X_test = test_df.drop(columns=['id'])

# 7.2 Retrain final model on full training data
final_model.fit(
    X, y,
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(100)]  # prints every 100 rounds, remove if you want silence
)

# 7.3 Predict probabilities for submission
test_preds = final_model.predict_proba(X_test)[:, 1]

# 7.4 Create submission file
submission = pd.DataFrame({"id": test_ids, "y": test_preds})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created: submission.csv")
submission.head()



# 8.1 Validation predictions from tuned models
valid_lgb = best_lgb.predict_proba(X_valid)[:, 1]
valid_xgb = best_xgb.predict_proba(X_valid)[:, 1]

# 8.2 Fast weight search on validation to maximize ROC-AUC
weights = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00
best_w, best_auc = None, -1.0
for w in weights:
    blend = w * valid_lgb + (1 - w) * valid_xgb
    auc = roc_auc_score(y_valid, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

print(f"Best prob-blend weight (LGBM): {best_w:.2f} | Val AUC: {best_auc:.5f}")

# 8.3 Rank-average (more robust to calibration)
rank_lgb = pd.Series(valid_lgb).rank(pct=True).values
rank_xgb = pd.Series(valid_xgb).rank(pct=True).values
rank_blend_val = 0.5 * rank_lgb + 0.5 * rank_xgb
rank_auc = roc_auc_score(y_valid, rank_blend_val)
print(f"Rank-average Val AUC: {rank_auc:.5f}")

# 8.4 Pick the better validation strategy
use_rank = rank_auc > best_auc
print("Using rank-average" if use_rank else f"Using prob-blend (w={best_w:.2f})")

# 8.5 Test predictions & final blend
test_lgb = best_lgb.predict_proba(X_test)[:, 1]
test_xgb = best_xgb.predict_proba(X_test)[:, 1]

if use_rank:
    test_pred = 0.5 * pd.Series(test_lgb).rank(pct=True).values + \
                0.5 * pd.Series(test_xgb).rank(pct=True).values
else:
    test_pred = best_w * test_lgb + (1 - best_w) * test_xgb

# 8.6 Create final submission (Kaggle picks files from /kaggle/working)
submission = pd.DataFrame({"id": test_ids, "y": test_pred})
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("âœ… Saved ensemble submission to:", out_path)
print(submission.head())


# --- Create submission in the Kaggle working directory ---
#submission = pd.DataFrame({"id": test_ids, "y": test_preds})

# sanity checks
#assert "id" in submission.columns and "y" in submission.columns, "Columns must be exactly ['id','y']"
#assert submission["id"].is_monotonic_increasing or submission["id"].is_monotonic_decreasing, "IDs look unorderedâ€”still ok, but FYI"
#assert submission["y"].between(0,1).all(), "y must be probabilities in [0,1]"

#out_path = "/kaggle/working/submission.csv"
#submission.to_csv(out_path, index=False)

# verify
#import os
#print("Saved to:", out_path)
#print("File size (bytes):", os.path.getsize(out_path))
#print(submission.head(3))

# list working dir so Kaggle picks it up as an Output artifact
#print("\nFiles in /kaggle/working:")
##rint(os.listdir("/kaggle/working"))




