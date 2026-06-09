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


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

# 1. データ読み込み
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# 2. 欠損値処理（数値は平均、カテゴリは最頻値）
num_cols = train.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in num_cols:
    mean_val = train[col].mean()
    train[col] = train[col].fillna(mean_val)
    if col in test.columns:
        test[col] = test[col].fillna(mean_val)

cat_cols = train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if col == "Personality":  # 目的変数はtestにないので除外
        train[col] = train[col].fillna(train[col].mode()[0])
        continue
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    if col in test.columns:
        test[col] = test[col].fillna(mode_val)

# 3. 目的変数エンコード
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])

# 4. 特徴量とラベルの準備
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])

# 5. カテゴリ変数をまとめてエンコード
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)
cat_cols_combined = combined.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
combined[cat_cols_combined] = encoder.fit_transform(combined[cat_cols_combined])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)

# 6. モデルパラメータ設定
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist"
}

# 7. クロスバリデーションで学習
N_SPLITS = 5
N_REPEATS = 1
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50,
                      verbose_eval=100)

    oof_preds[val_idx] += model.predict(dval) / N_REPEATS
    test_preds += model.predict(dtest) / (N_REPEATS * N_SPLITS)

# 8. 評価指標の表示
ll = log_loss(y, oof_preds)
cv_acc = accuracy_score(y, oof_preds > 0.5)
print(f"Cross-Validation log loss: {ll:.4f}, accuracy: {cv_acc:.4f}")

# 9. 提出用ファイル作成
final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print(submission.head())


# 3. 欠損値補完（平均・最頻値）
num_cols = train.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in num_cols:
    mean_val = train[col].mean()
    train[col] = train[col].fillna(mean_val)
    if col in test.columns:
        test[col] = test[col].fillna(mean_val)

cat_cols = train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if col == "Personality":  # 目的変数はtestにないので除外
        train[col] = train[col].fillna(train[col].mode()[0])
        continue
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    if col in test.columns:
        test[col] = test[col].fillna(mode_val)

# 4. 目的変数エンコード
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])


# 5. 特徴量と目的変数設定
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])

# 6. カテゴリ変数エンコード（学習＋テスト一括）
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)
cat_cols_combined = combined.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
combined[cat_cols_combined] = encoder.fit_transform(combined[cat_cols_combined])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


# 7. モデルパラメータ設定
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist"  # Kaggle環境で高速化
}

# 8. クロスバリデーション設定
N_SPLITS = 5
N_REPEATS = 1
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50,
                      verbose_eval=100)

    oof_preds[val_idx] += model.predict(dval) / N_REPEATS
    test_preds += model.predict(dtest) / (N_REPEATS * N_SPLITS)


# 9. 評価表示
ll = log_loss(y, oof_preds)
cv_acc = accuracy_score(y, oof_preds > 0.5)
print(f"Cross-Validation log loss: {ll:.4f}, accuracy: {cv_acc:.4f}")

# 10. 重要特徴量（Gain）表示（任意）
importance = model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("Top features by gain:")
for f, g in importance[:10]:
    print(f"{f}: {g:.3f}")

# 11. 提出ファイル作成
final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print(submission.head())


import os

model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # (前略) 学習部分は省略
    
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=50,
                      verbose_eval=100)

    # モデルをファイル保存（JSON形式がおすすめ）
    model_path = os.path.join(model_dir, f"xgb_model_fold{fold}.json")
    model.save_model(model_path)

    # (以降省略)

