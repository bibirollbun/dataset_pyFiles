import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import lightgbm as lgb



train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

TARGET = "diagnosed_diabetes"
ID_COL = "id"

y = train[TARGET]
train_ids = train[ID_COL]
test_ids  = test[ID_COL]

train.drop([TARGET], axis=1, inplace=True)



def add_features(df):
    df = df.copy()
    df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
    df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["activity_per_age"] = df["physical_activity_minutes_per_week"] / (df["age"] + 1)
    df["sleep_screen"] = df["sleep_hours_per_day"] - df["screen_time_hours_per_day"]
    return df

train = add_features(train)
test  = add_features(test)

# Missing indicators
for col in train.columns:
    if train[col].isnull().sum() > 0:
        train[col + "_isna"] = train[col].isnull().astype(int)
        test[col + "_isna"]  = test[col].isnull().astype(int)

# Interactions
train["age_bmi"] = train["age"] * train["bmi"]
test["age_bmi"]  = test["age"]  * test["bmi"]



cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

def target_encode(train, test, y, cols, n_splits=5, seed=42):
    train_te = train.copy()
    test_te  = test.copy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y.mean()

    for col in cols:
        train_te[col + "_te"] = 0.0

        for tr_idx, val_idx in skf.split(train, y):
            means = pd.concat(
                [train.iloc[tr_idx][col], y.iloc[tr_idx]], axis=1
            ).groupby(col)[y.name].mean()

            train_te.loc[val_idx, col + "_te"] = train.iloc[val_idx][col].map(means)

        full_means = pd.concat([train[col], y], axis=1).groupby(col)[y.name].mean()
        test_te[col + "_te"] = test[col].map(full_means)

        train_te[col + "_te"].fillna(global_mean, inplace=True)
        test_te[col + "_te"].fillna(global_mean, inplace=True)

        train_te.drop(col, axis=1, inplace=True)
        test_te.drop(col, axis=1, inplace=True)

    return train_te, test_te

train, test = target_encode(train, test, y, cat_cols)






SEEDS = [42, 2024]
FOLDS = 5

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)



test_lgbm_seeds = []

for seed in SEEDS:
    print(f"\nðŸ”¥ LGBM seed {seed}")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.025,
        "n_estimators": 2000,
        "num_leaves": 48,
        "max_depth": 7,
        "min_child_samples": 150,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.0,
        "reg_lambda": 4.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1
    }

    for tr_idx, val_idx in skf.split(train, y):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred   += model.predict_proba(test)[:, 1] / FOLDS

    print("CV AUC:", roc_auc_score(y, oof))
    test_lgbm_seeds.append(test_pred)



test_xgb_seeds = []

for seed in SEEDS:
    print(f"\nðŸ”¥ XGB seed {seed}")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    params = {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": seed
    }

    for tr_idx, val_idx in skf.split(train, y):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred   += model.predict_proba(test)[:, 1] / FOLDS

    print("CV AUC:", roc_auc_score(y, oof))
    test_xgb_seeds.append(test_pred)



def sharpen(p, power=1.25):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p ** power) / ((p ** power) + ((1 - p) ** power))



lgbm_avg = (test_lgbm_seeds[0] + test_lgbm_seeds[1]) / 2
xgb_avg  = (test_xgb_seeds[0]  + test_xgb_seeds[1])  / 2

lgbm_sharp = sharpen(lgbm_avg, power=1.25)

final_test = (
    0.70 * lgbm_sharp +
    0.30 * xgb_avg
)



np.random.seed(42)
final_test += np.random.normal(
    0, 0.00015, len(final_test)
) * (final_test - final_test.mean())



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_test
})

submission.to_csv("submission_final_push.csv", index=False)
submission.head()





