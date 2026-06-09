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

# データの読み込み
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# （確認のため）先頭5行を表示
train.head()


# 数値の列は平均で埋める
num_cols = train.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in num_cols:
    mean_val = train[col].mean()
    train[col] = train[col].fillna(mean_val)
    if col in test.columns:
        test[col] = test[col].fillna(mean_val)

# カテゴリの列（object型）は最頻値で埋める
cat_cols = train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if col == "Personality":  # ラベル列は test に存在しないのでスキップ
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        continue
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    if col in test.columns:
        test[col] = test[col].fillna(mode_val)

# 欠損が残っていないか確認
print("Train の欠損数：")
print(train.isnull().sum())
print("\nTest の欠損数：")
print(test.isnull().sum())


from sklearn.preprocessing import OrdinalEncoder

# 学習データとテストデータを一緒にして変換（同じルールで変換するため）
combined = pd.concat([train.drop(columns=["Personality"]), test], axis=0).reset_index(drop=True)

# object型（文字）の列を探す
cat_cols = combined.select_dtypes(include="object").columns.tolist()

# エンコーダで数値化
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

# 分割し直す（元のtrain/testの長さに戻す）
X = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):].reset_index(drop=True)

# ラベル（目的変数）を数値に変換（Extrovert→1, Introvert→0）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train["Personality"])


print(X.head())     # 変換された学習データ（最初の5行）
print(y[:5])        # 目的変数（ラベル）の最初の5つ
print(X_test.head())  # 変換されたテストデータ（最初の5行）


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# XGBoost 用にデータを変換
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)

# パラメータ設定
params = {
    "objective": "binary:logistic",  # 2値分類
    "eval_metric": "logloss",        # ロス関数（小さいほど良い）
    "max_depth": 4,                  # 木の深さ
    "eta": 0.05,                     # 学習率
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# クロスバリデーションでモデル学習
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain_fold = xgb.DMatrix(X_train, label=y_train)
    dval_fold = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain_fold, num_boost_round=1000,
                      evals=[(dval_fold, "validation")],
                      early_stopping_rounds=30,
                      verbose_eval=False)

    oof_preds[val_idx] = model.predict(dval_fold)
    test_preds += model.predict(dtest) / skf.n_splits

# 精度確認
print("Log Loss:", log_loss(y, oof_preds))
print("Accuracy:", accuracy_score(y, oof_preds > 0.5))


# しきい値0.5で二値分類の予測結果を決定
final_preds = (test_preds > 0.5).astype(int)

# ラベルを元の文字列（Introvert / Extrovert）に戻す
submission["Personality"] = le.inverse_transform(final_preds)

# submission.csvとして保存（提出用ファイル）
submission.to_csv("submission.csv", index=False)

# ファイルの先頭を確認（任意）
print(submission.head())

