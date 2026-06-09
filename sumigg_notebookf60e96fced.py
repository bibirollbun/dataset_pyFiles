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

# 先頭5行を表示
train.head()


import pandas as pd

# 数値列とカテゴリ列の取得
num_cols = train.select_dtypes(include=["number"]).columns
cat_cols = train.select_dtypes(include="object").columns

# 数値列の欠損を平均値で埋める
train[num_cols] = train[num_cols].apply(lambda col: col.fillna(col.mean()))
test[num_cols] = test[num_cols].apply(lambda col: col.fillna(train[col.name].mean()) if col.name in train.columns else col)

# ラベル列 'Personality' を除いたカテゴリ列の欠損を最頻値で埋める
for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    if col != "Personality" and col in test.columns:
        test[col] = test[col].fillna(mode_val)

# 欠損の確認出力
print("Train の欠損数：\n", train.isnull().sum())
print("\nTest の欠損数：\n", test.isnull().sum())


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ラベル列を別で保存しておく
labels = train["Personality"].copy()

# Personality列を除いたtrainとtestを結合（特徴量だけまとめる）
features_train = train.drop("Personality", axis=1)
combined_df = pd.concat([features_train, test], ignore_index=True)

# カテゴリ変数（文字列型）を取得
categorical_columns = combined_df.select_dtypes(include=["object"]).columns

# OrdinalEncoderでカテゴリを数値に変換
ordinal_encoder = OrdinalEncoder()
combined_df[categorical_columns] = ordinal_encoder.fit_transform(combined_df[categorical_columns])

# 結合したデータを再分割（学習用・テスト用に戻す）
X = combined_df.iloc[:len(train)].reset_index(drop=True)
X_test = combined_df.iloc[len(train):].reset_index(drop=True)

# ラベル（Extrovert / Introvert）を数値に変換（Extrovert→1, Introvert→0）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


print(X.head())     # 変換された学習データ
print(y[:5])        # 目的変数（ラベル）の最初の5つ
print(X_test.head())  # 変換されたテストデータ


import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

# XGBoost 用のデータセット作成（X_test は全 fold 共通）
dtest = xgb.DMatrix(X_test)

# モデルのハイパーパラメータ設定
xgb_params = {
    "objective": "binary:logistic",   # 2クラス分類
    "eval_metric": "logloss",         # 評価指標（損失関数）
    "max_depth": 4,                   # 決定木の深さ
    "eta": 0.05,                      # 学習率（小さいほどゆっくり学習）
    "subsample": 0.8,                 # データのサブサンプリング
    "colsample_bytree": 0.8,          # 特徴量のサブサンプリング
    "random_state": 42                # 乱数シード（再現性のため）
}

# クロスバリデーション設定（Stratified で層化抽出）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 予測用配列を用意
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

# 各 fold で学習と予測
for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost 用 DMatrix に変換
    dtrain_fold = xgb.DMatrix(X_tr, label=y_tr)
    dvalid_fold = xgb.DMatrix(X_val, label=y_val)

    # モデル学習（early stopping 付き）
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain_fold,
        num_boost_round=1000,
        evals=[(dvalid_fold, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # バリデーションとテストの予測
    oof_predictions[val_idx] = booster.predict(dvalid_fold)
    test_predictions += booster.predict(dtest) / cv.n_splits

# 評価指標を出力
logloss_score = log_loss(y, oof_predictions)
accuracy = accuracy_score(y, (oof_predictions > 0.5).astype(int))

print(f"Log Loss: {logloss_score:.5f}")
print(f"Accuracy: {accuracy:.5f}")


# 予測確率に対して閾値0.5を使い、最終的なラベル（0 or 1）を決定
predicted_labels = (test_predictions > 0.5).astype(int)

# 数値ラベル（0 or 1）を元のカテゴリ名（"Introvert"/"Extrovert"）に戻す
submission["Personality"] = label_encoder.inverse_transform(predicted_labels)


print("特徴量（説明変数）の数：", len(X.columns))
print("特徴量一覧：")
print(X.columns.tolist())


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# パラメータ（改良版）
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "lambda": 1.0,
    "alpha": 0.5,
    "random_state": 42,
    "tree_method": "hist",  # GPU使用環境なら "gpu_hist"
}

# クロスバリデーション
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# しきい値チューニングのために保存
val_thresholds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain_fold = xgb.DMatrix(X_train, label=y_train)
    dval_fold = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain_fold, num_boost_round=3000,
                      evals=[(dval_fold, "validation")],
                      early_stopping_rounds=100,
                      verbose_eval=100)

    val_preds = model.predict(dval_fold)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(xgb.DMatrix(X_test)) / skf.n_splits

    # 閾値調整（各foldでベストなものを記録）
    thresholds = np.arange(0.3, 0.7, 0.01)
    accs = [accuracy_score(y_val, val_preds > thr) for thr in thresholds]
    best_thr = thresholds[np.argmax(accs)]
    val_thresholds.append(best_thr)
    print(f"  Best threshold: {best_thr:.2f} | Accuracy: {max(accs):.4f}")

# 全体スコア
mean_thr = np.mean(val_thresholds)
print(f"\nBest threshold (avg): {mean_thr:.3f}")
print("Final Log Loss:", log_loss(y, oof_preds))
print("Final Accuracy:", accuracy_score(y, oof_preds > mean_thr))

# 特徴量重要度（最後のモデルで表示）
xgb.plot_importance(model, max_num_features=20, importance_type='gain', height=0.5)
plt.show()

# 結果を CSV ファイルとして保存（提出用）
submission.to_csv("submission.csv", index=False)

# ファイルの先頭を表示して確認（任意）
print(submission.head())

