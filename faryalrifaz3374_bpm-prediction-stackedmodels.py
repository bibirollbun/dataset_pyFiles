# Importing libraries
import pandas as pd
import numpy as np

# File paths (already provided by Kaggle environment)
train_path = "/kaggle/input/playground-series-s5e9/train.csv"
test_path = "/kaggle/input/playground-series-s5e9/test.csv"
sample_path = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

# Load datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

# Display first few rows
train.head()



# Check basic info
train.info()

# Check for missing values
train.isnull().sum()

# Show shape
print("Train shape:", train.shape)
print("Test shape:", test.shape)



# Display column names
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Check target variable
target_col = "target"  # Usually competitions have this, confirm from train
if target_col in train.columns:
    print("Target variable:", target_col)



# Missing values count per column for train dataset
missing_count_train = train.isnull().sum()

# Percentage of missing values
missing_percentage_train = (missing_count_train / len(train)) * 100

# Combine into one DataFrame
missing_train = pd.DataFrame({
    'Missing Values': missing_count_train,
    'Percentage': missing_percentage_train
})

missing_train



# Descriptive statistics for numerical columns
train.describe().T



import matplotlib.pyplot as plt
import seaborn as sns

train.drop(columns=['id']).hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,8))
sns.heatmap(train.drop(columns=['id']).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()



# Separate boxplots for each column (excluding 'id')
for col in train.columns.drop('id'):
    plt.figure(figsize=(6,3))
    sns.boxplot(x=col, data=train, orient='h')
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()


# Plot train and test separately for common features (exclude 'id')
common_cols = [c for c in train.columns if c in test.columns and c != 'id']

for c in common_cols:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.kdeplot(train[c], ax=axes[0], fill=True, color='C0')
    axes[0].set_title(f"Train: {c}")
    sns.kdeplot(test[c], ax=axes[1], fill=True, color='C1')
    axes[1].set_title(f"Test: {c}")
    plt.tight_layout()
    plt.show()

# If a target exists only in train (e.g., BeatsPerMinute), plot it separately
if 'BeatsPerMinute' in train.columns and 'BeatsPerMinute' not in test.columns:
    plt.figure(figsize=(6,4))
    sns.kdeplot(train['BeatsPerMinute'], fill=True, color='C0')
    plt.title("Train only: BeatsPerMinute")
    plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from xgboost import XGBRegressor

SEED = 42
FOLDS = 5

id_col = sample_submission.columns[0]
target_col = sample_submission.columns[1]

# Features
features = [c for c in train.columns if c not in [id_col, target_col]]

# Feature engineering
def add_features(df):
    out = df.copy()
    out["DurationMin"] = out["TrackDurationMs"] / 60000.0
    out["Energy_x_Rhythm"] = out["Energy"] * out["RhythmScore"]
    out["Vocal_to_Instrumental"] = out["VocalContent"] / (out["InstrumentalScore"] + 1e-6)
    out["Acoustic_x_Instrumental"] = out["AcousticQuality"] * out["InstrumentalScore"]
    out["Loudness_abs"] = out["AudioLoudness"].abs()
    out["Energy_sq"] = out["Energy"] ** 2
    out["MoodScore_sq"] = out["MoodScore"] ** 2
    out["Live_x_Mood"] = out["LivePerformanceLikelihood"] * out["MoodScore"]
    return out

X = add_features(train[features])
y = train[target_col].astype(float)
X_test = add_features(test[features])

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# Out of fold predictions for stacking
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
meta_test_lgb = np.zeros(len(X_test))
meta_test_xgb = np.zeros(len(X_test))

lgb_params = dict(
    n_estimators=20000,
    learning_rate=0.003,
    num_leaves=127,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
)
xgb_params = dict(
    n_estimators=20000,
    learning_rate=0.003,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration_)
    meta_test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_) / FOLDS

    # XGBoost
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        early_stopping_rounds=200,
        verbose=False,
    )
    best_iter = getattr(xgb_model, "best_iteration", None)
    if best_iter is not None:
        oof_xgb[va_idx] = xgb_model.predict(X_va, iteration_range=(0, best_iter + 1))
        meta_test_xgb += xgb_model.predict(X_test, iteration_range=(0, best_iter + 1)) / FOLDS
    else:
        oof_xgb[va_idx] = xgb_model.predict(X_va)
        meta_test_xgb += xgb_model.predict(X_test) / FOLDS

    fold_rmse = mean_squared_error(y_va, 0.5*oof_lgb[va_idx] + 0.5*oof_xgb[va_idx], squared=False)
    print(f"Fold {fold} RMSE (0.5 blend): {fold_rmse:.5f}")

# Stacking meta-model
meta_X = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb})
meta_X_test = pd.DataFrame({'lgb': meta_test_lgb, 'xgb': meta_test_xgb})

stacker = Ridge(alpha=3.0)
stacker.fit(meta_X, y)
stacked_pred = stacker.predict(meta_X_test)

# Safe clipping
stacked_pred = np.clip(stacked_pred, np.percentile(y, 0.5), np.percentile(y, 99.5))

print(f"\nOOF RMSE LGBM:   {mean_squared_error(y, oof_lgb, squared=False):.5f}")
print(f"OOF RMSE XGBoost:{mean_squared_error(y, oof_xgb, squared=False):.5f}")
print(f"OOF RMSE STACKED: {mean_squared_error(y, stacker.predict(meta_X), squared=False):.5f}")

# Save submission
submission = sample_submission.copy()
submission[target_col] = stacked_pred
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv, head:")
print(submission.head())

