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


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
import lightgbm as lgb
import json

import seaborn as sns
import matplotlib.pyplot as plt



train_data = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train_df = train_data.drop(columns=['id'])
train_df.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test = test_data.drop(columns=["id"])


train_df.info()


train_df.describe().T


cat_cols = train_df.drop(columns="diagnosed_diabetes").select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train_df.drop(columns="diagnosed_diabetes").select_dtypes(include=["int64", "float64"]).columns.tolist()


for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col)
    plt.title(f'Distribution of {col.capitalize()}')
    plt.show()


for col in num_cols:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1])
    
    # --- Histogram ---
    sns.histplot(data=train_df, x=col, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of {col}', fontsize=14)
    axes[0].set_xlabel('')  # remove x label to save space
    
    # --- Boxplot ---
    sns.boxplot(data=train_df, x=col, ax=axes[1], color='skyblue')
    axes[1].set_xlabel(col, fontsize=12)
    
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 6))
plt.pie(
    train_df["diagnosed_diabetes"].value_counts(),
    labels=train_df["diagnosed_diabetes"].value_counts().index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Target Distribution")
plt.show()


train_df["diagnosed_diabetes"].value_counts(normalize=True)


def add_features(df):
    df = df.copy()

    df["chol_hdl_ratio"] = df["cholesterol_total"] / df["hdl_cholesterol"]
    df["tg_hdl_ratio"] = df["triglycerides"] / df["hdl_cholesterol"]
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["bmi_age"] = df["bmi"] * df["age"]
    df["activity_per_bmi"] = df["physical_activity_minutes_per_week"] / df["bmi"]

    return df


train = add_features(train_df)
test = add_features(test)


target = "diagnosed_diabetes"
y = train[target]
X = train.drop(columns=[target])


categorical_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")
cat_idx = [X.columns.get_loc(c) for c in categorical_cols]


def optimize_catboost(X, y, cat_features_idx, n_trials=50):
    
    def objective(trial):

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": trial.suggest_int("iterations", 800, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1, 20),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "task_type": "GPU",
            "verbose": False
        }

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_features_idx)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best CatBoost Params:", study.best_params)
    print("Best AUC:", study.best_value)

    return study.best_params


def optimize_lgbm(X, y, n_trials=50, categorical_cols=None):

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 600, 3000),
            "device": "gpu"
        }

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
        valid_dataset = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols)

        model = lgb.train(params, train_dataset, valid_sets=[valid_dataset])

        preds = model.predict(X_valid)
        auc = roc_auc_score(y_valid, preds)

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best LGBM Params:", study.best_params)
    print("Best AUC:", study.best_value)

    return study.best_params


#best_cat_params = optimize_catboost(X, y, cat_idx, n_trials=30)


#best_lgbm_params = optimize_lgbm(X, y, n_trials=30, categorical_cols=categorical_cols)


with open("/kaggle/input/best-params/best_cat_params.json", "r") as f:
    best_cat_params = json.load(f)

with open("/kaggle/input/best-params/best_lgbm_params.json", "r") as f:
    best_lgbm_params = json.load(f)


best_lgbm_params.update({"verbose_eval": -1})


kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train))
preds_cat = np.zeros(len(test))


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool  = Pool(test, cat_features=cat_idx)  

    model = CatBoostClassifier(**best_cat_params)

    model.fit(train_pool, eval_set=valid_pool)

    # Save OOF preds
    oof_cat[val_idx] = model.predict_proba(X_val)[:,1]
    preds_cat += model.predict_proba(test)[:,1] / kf.n_splits


oof_lgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"LGBM Fold {fold+1}/10")

    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    train_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx)
    valid_lgb = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_idx)


    model = lgb.train(best_lgbm_params, train_lgb, num_boost_round=3000, valid_sets=[valid_lgb])

    oof_lgb[val_idx] = model.predict(X_val)
    preds_lgb += model.predict(test) / 10


# In order to save best parameters.
#with open("best_cat_params.json", "w") as f:
#    json.dump(best_cat_params, f, indent=4)

#with open("best_lgbm_params.json", "w") as f:
#    json.dump(best_lgbm_params, f, indent=4)


oof_final = 0.6 * oof_cat + 0.4 * oof_lgb
test_final = 0.6 * preds_cat + 0.4 * preds_lgb

print("CatBoost OOF AUC:", roc_auc_score(y, oof_cat))
print("LGBM OOF AUC:", roc_auc_score(y, oof_lgb))
print("Ensemble OOF AUC:", roc_auc_score(y, oof_final))


submission = pd.DataFrame({
    "id": test_data["id"],
    "diagnosed_diabetes": test_final
})

submission.to_csv("submission.csv", index=False)

