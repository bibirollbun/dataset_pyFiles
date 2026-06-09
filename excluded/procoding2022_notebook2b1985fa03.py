import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import combinations
from lightgbm import early_stopping, log_evaluation

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.simplefilter('ignore')

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Encode binary categorical feature
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# Numerical features
num_feats = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# Feature Engineering
def engineer_features(df):
    # Basic Interactions
    for f1, f2 in combinations(num_feats, 2):
        df[f"{f1}_x_{f2}"] = df[f1] * df[f2]

    # Derived and ratio features
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["HR_per_min"] = df["Heart_Rate"] / (df["Duration"] + 1e-5)
    df["Temp_x_HR"] = df["Body_Temp"] * df["Heart_Rate"]
    df["Weight_per_Age"] = df["Weight"] / (df["Age"] + 1e-5)

    # Polynomial features
    for col in num_feats:
        df[f"{col}_squared"] = df[col] ** 2
        df[f"{col}_log"] = np.log1p(df[col])

    return df

train = engineer_features(train)
test = engineer_features(test)

# Scaling
scaler = StandardScaler()
all_features = train.drop(columns=["id", "Calories"]).columns

train[all_features] = scaler.fit_transform(train[all_features])
test[all_features] = scaler.transform(test[all_features])

# Prepare data
X = train[all_features]
y = np.log1p(train["Calories"])
X_test = test[all_features]

# Cross-validation
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

# Models
def get_xgb():
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=1.2,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

def get_lgb():
    return LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1
    )

# Training Loop
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸ“‚ Fold {fold+1}")

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # XGB
    model_xgb = get_xgb()
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=0
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / FOLDS

    # LGB
    model_lgb = get_lgb()
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(100),
            log_evaluation(0)
        ]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / FOLDS

    # Fold Scores
    rmse_x = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    rmse_l = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"âœ… Fold {fold+1} XGB RMSE: {rmse_x:.5f} | LGB RMSE: {rmse_l:.5f}")

# Final Scores
print(f"\nğŸ�� XGB Final CV RMSE: {np.sqrt(mean_squared_error(y, oof_xgb)):.5f}")
print(f"ğŸ�� LGB Final CV RMSE: {np.sqrt(mean_squared_error(y, oof_lgb)):.5f}")

# Weighted Ensemble
final_preds = 0.5 * np.expm1(test_xgb) + 0.5 * np.expm1(test_lgb)
final_preds = np.clip(final_preds, train["Calories"].min(), train["Calories"].max())

# Submission
submission["Calories"] = final_preds
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission Preview:")
print(submission.head())

