import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


SEED = np.random.randint(0,256)
N_FOLDS = 5
TARGET = "accident_risk"

TRAIN_FILE = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_FILE = "/kaggle/input/playground-series-s5e10/test.csv"
SUBMISSION_FILE = "/kaggle/input/playground-series-s5e10/sample_submission.csv"


print(f'Train shape: {pd.read_csv(TRAIN_FILE).shape}')
print(f'Test shape: {pd.read_csv(TEST_FILE).shape}')

print(pd.read_csv(TEST_FILE).info())


plt.figure(figsize=(6,4))
sns.histplot(pd.read_csv(TRAIN_FILE)['accident_risk'], bins=30, kde=True)
plt.title('Distribution of Target (accident_risk)')
plt.xlabel('accident_risk')
plt.ylabel('Frequency')
plt.show()


num_features = pd.read_csv(TRAIN_FILE).select_dtypes(include=['int64','float64']).columns.tolist()
corr = pd.read_csv(TRAIN_FILE)[num_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Numerical Features)')
plt.show()


cat_features = pd.read_csv(TRAIN_FILE).select_dtypes(include=['object','bool']).columns.tolist()
for col in cat_features:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=col, y='accident_risk', data=pd.read_csv(TRAIN_FILE))
    plt.title(f'{col} vs accident_risk')
    plt.xticks(rotation=45)
    plt.show()



pd.read_csv(TRAIN_FILE)[num_features].plot(kind='box', subplots=True, layout=(3,3), figsize=(12,8), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


def read_data():
    for fname in [TRAIN_FILE, "TRAIN.csv", "train.csv", "train_data.csv"]:
        if os.path.exists(fname):
            train = pd.read_csv(fname)
            break
    else:
        raise FileNotFoundError("Train file not found - put train.csv in working directory or change TRAIN_FILE")

    for fname in [TEST_FILE, "TEST.csv", "test.csv", "test_data.csv"]:
        if os.path.exists(fname):
            test = pd.read_csv(fname)
            break
    else:
        raise FileNotFoundError("Test file not found - put test.csv in working directory or change TEST_FILE")

    return train, test

def basic_feature_engineering(df):
    for col in df.columns:
        if df[col].dtype == object:
            if set(df[col].dropna().unique()).issubset({"True","False","true","false","TRUE","FALSE"}):
                df[col] = df[col].map({"True":1,"False":0,"true":1,"false":0,"TRUE":1,"FALSE":0})
    if "curvature" in df.columns and "speed_limit" in df.columns:
        df["curv_speed"] = df["curvature"] * df["speed_limit"]
    if "num_lanes" in df.columns and "speed_limit" in df.columns:
        df["lanes_per_speed"] = df["num_lanes"] / (df["speed_limit"] + 1e-6)
    return df

def preprocess(train, test):
    test_ids = test["id"].copy()
    train = train.copy()
    test = test.copy()

    if TARGET not in train.columns:
        raise KeyError(f"{TARGET} not in train")

    train = basic_feature_engineering(train)
    test = basic_feature_engineering(test)

    ignore_cols = ["id", TARGET]
    features = [c for c in train.columns if c not in ignore_cols]
    num_cols = train[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    for c in num_cols:
        if train[c].isnull().any() or test[c].isnull().any():
            med = train[c].median()
            train[c].fillna(med, inplace=True)
            test[c].fillna(med, inplace=True)

    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        combined = pd.concat([train[cat_cols], test[cat_cols]], axis=0)
        enc.fit(combined)
        train_cat = pd.DataFrame(enc.transform(train[cat_cols]), columns=cat_cols, index=train.index)
        test_cat = pd.DataFrame(enc.transform(test[cat_cols]), columns=cat_cols, index=test.index)
        train[cat_cols] = train_cat
        test[cat_cols] = test_cat

    final_features = [c for c in train.columns if c not in ["id", TARGET]]
    return train, test, final_features, test_ids

def fit_lgb(train_X, train_y, val_X, val_y):
    train_set = lgb.Dataset(train_X, train_y)
    val_set = lgb.Dataset(val_X, val_y, reference=train_set)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": SEED,
        "verbosity": -1,
        "n_jobs": -1
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        callbacks = [
            lgb.early_stopping(100),
            lgb.log_evaluation(200)
        ]
    )
    return model

def fit_xgb(train_X, train_y, val_X, val_y):
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dval = xgb.DMatrix(val_X, label=val_y)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "seed": SEED,
        "nthread": 8
    }
    evallist = [(dtrain, "train"), (dval, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=evallist,
        early_stopping_rounds=100,
        verbose_eval=200
    )
    return model

def run_training(train, test, features, test_ids):
    X = train[features].reset_index(drop=True)
    y = train[TARGET].values
    X_test = test[features].reset_index(drop=True)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    preds_lgb = np.zeros(len(X_test))
    preds_xgb = np.zeros(len(X_test))

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
        X_tr, X_val = X.loc[tr_idx], X.loc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        print("Training LightGBM...")
        lgb_model = fit_lgb(X_tr, y_tr, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        preds_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_FOLDS
        del lgb_model
        gc.collect()

        print("Training XGBoost...")
        xgb_model = fit_xgb(X_tr, y_tr, X_val, y_val)

        oof_xgb[val_idx] = xgb_model.predict(
            xgb.DMatrix(X_val),
            iteration_range=(0, xgb_model.best_iteration + 1)
        )
        preds_xgb += xgb_model.predict(
            xgb.DMatrix(X_test),
            iteration_range=(0, xgb_model.best_iteration + 1)
        ) / N_FOLDS

        del xgb_model
        gc.collect()

        fold_rmse_lgb = mean_squared_error(y_val, oof_lgb[val_idx], squared=False)
        fold_rmse_xgb = mean_squared_error(y_val, oof_xgb[val_idx], squared=False)
        print(f"Fold {fold+1} LGB RMSE: {fold_rmse_lgb:.6f}")
        print(f"Fold {fold+1} XGB RMSE: {fold_rmse_xgb:.6f}")

    X_meta = np.vstack([oof_lgb, oof_xgb]).T
    X_test_meta = np.vstack([preds_lgb, preds_xgb]).T

    print("\nTraining Ridge meta-model on OOF predictions...")
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    meta = Ridge(alpha=1.0, random_state=SEED)
    meta.fit(X_meta_scaled, y)
    final_oof = meta.predict(X_meta_scaled)
    cv_rmse = mean_squared_error(y, final_oof, squared=False)
    print(f"\nCV RMSE (stacked): {cv_rmse:.6f}")

    final_preds = meta.predict(X_test_meta_scaled)
    final_preds = np.clip(final_preds, 0.0, 1.0)

    return final_preds, cv_rmse

def main():
    t0 = time.time()
    train, test = read_data()
    print("Loaded data. Rows train:", len(train), "test:", len(test))
    train_proc, test_proc, features, test_ids = preprocess(train, test)
    print("Features used:", len(features))
    preds, cv_rmse = run_training(train_proc, test_proc, features, test_ids)

    submission = pd.DataFrame({"id": test_ids, "accident_risk": preds})
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission to submission.csv CV RMSE: {cv_rmse:.6f}")
    print("Elapsed:", round(time.time() - t0, 1), "s")

if __name__ == "__main__":
    main()

