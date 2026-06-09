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
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef


df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# 提出用にテストデータのIDを保持
test_id = test_df['id']

# ID列を削除
df = df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# ===== ステップ3: ラベルエンコーディング =====
print("ラベルエンコーディングを開始します...")
# 'class'列を除いた、すべての文字データ列を対象にする
feature_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'class']

for col in feature_cols:
    # 訓練データとテストデータを結合し、すべてのカテゴリをエンコーダーに学習させる
    combined_series = pd.concat([df[col], test_df[col]]).astype(str)
    
    le = LabelEncoder()
    le.fit(combined_series)
    
    # 訓練データとテストデータをそれぞれ変換
    df[col] = le.transform(df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# class列も数値に変換 (e:0, p:1)
df['class'] = df['class'].map({'e': 0, 'p': 1})

print("エンコーディングが完了しました。")


# ===== ステップ4: Optunaによるハイパーパラメータチューニング =====
# 特徴量Xと目的変数yを定義
X = df.drop('class', axis=1)
y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.01,
    'n_estimators': 1097,
    'num_leaves': 956,
    'min_child_samples': 112,
    'reg_alpha': 0.19812645495932385,
    'reg_lambda': 0.34036226127746694,
    'colsample_bytree': 0.43199251589983806,
    'random_state': 42,
    'verbose': -1
}


# モデルを作成
model = lgb.LGBMClassifier(**lgbm_params)
# 訓練データの一部(X_train)で学習
print("モデルの学習を開始します...")
model.fit(X_train, y_train)
print("学習が完了しました。")


# 検証用データ(X_val)で性能を評価
print("\n--- ローカルでの性能評価 ---")
predictions_local = model.predict(X_val)
mcc_local = matthews_corrcoef(y_val, predictions_local)
print(f"ローカルでのMCCスコア: {mcc_local:.5f}")


# ===== ステップ3: 提出用モデルの作成と予測 =====
# 全ての訓練データを使って再学習し、モデルの性能を最大化
print("\n全データで最終モデルを再学習します...")
# verboseを1にすると学習過程が表示される
lgbm_params['verbose'] = 1 
final_model = lgb.LGBMClassifier(**lgbm_params)
final_model.fit(X, y)
print("再学習が完了しました。")


# テストデータで予測
predictions_submission = final_model.predict(test_df)

# 目的変数のマッピングを元に戻す
class_map_reverse = {0: 'e', 1: 'p'}
predictions_labels = [class_map_reverse[pred] for pred in predictions_submission]

# 提出用ファイルの作成
submission_df = pd.DataFrame({'id': test_id, 'class': predictions_labels})
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' が作成されました。")
print(submission_df.head())

