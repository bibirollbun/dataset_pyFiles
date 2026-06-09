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
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# --- ステップ0: ファイルパスの自動検出とデータの読み込み ---
# Kaggleの/kaggle/input/ディレクトリを探索して、必要なファイルのフルパスを自動的に見つけます。
train_path = ''
test_path = ''
submission_path = ''

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'train.csv' in filename:
            train_path = os.path.join(dirname, filename)
        elif 'test.csv' in filename:
            test_path = os.path.join(dirname, filename)
        elif 'sample_submission.csv' in filename:
            submission_path = os.path.join(dirname, filename)

# 見つけたパスを使って、3つのファイルを読み込みます
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(submission_path)

print("データの読み込みが完了しました。")
print(f"訓練データパス: {train_path}")
print(f"テストデータパス: {test_path}")
print(f"提出ファイルパス: {submission_path}")


# --- ステップ1: 欠損値の処理 ---
print("\n--- 欠損値の処理を開始 ---")
# 数値とカテゴリの列を特定 (idとターゲットは除外)
numeric_cols = train_df.select_dtypes(include=np.number).columns.drop(['id'])
categorical_cols = train_df.select_dtypes(include='object').columns.drop(['Personality'])

# 訓練データから補完値を計算（数値は中央値、カテゴリは最頻値）
imputation_values = {}
for col in numeric_cols:
    imputation_values[col] = train_df[col].median()
for col in categorical_cols:
    imputation_values[col] = train_df[col].mode()[0]

# 訓練データとテストデータの欠損値を補完
for col, value in imputation_values.items():
    train_df[col].fillna(value, inplace=True)
    test_df[col].fillna(value, inplace=True)
print("欠損値の処理が完了しました。")


# --- ステップ2: カテゴリカル変数のエンコーディング ---
print("\n--- エンコーディングを開始 ---")
# マッピング辞書を作成
binary_map = {'No': 0, 'Yes': 1}
personality_map = {'Extrovert': 0, 'Introvert': 1}

# 訓練データとテストデータのカテゴリカル変数を数値に変換
train_df['Stage_fear'] = train_df['Stage_fear'].map(binary_map)
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map(binary_map)
test_df['Stage_fear'] = test_df['Stage_fear'].map(binary_map)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(binary_map)

# 訓練データのターゲット変数(Personality)を数値に変換
train_df['Personality'] = train_df['Personality'].map(personality_map)
print("エンコーディングが完了しました。")


# --- ステップ3: モデルの学習と予測 ---
print("\n--- モデルの学習と予測を開始 ---")
# 訓練データから特徴量 (X) とターゲット (y) を分離
features = [col for col in train_df.columns if col not in ['id', 'Personality']]
X_train = train_df[features]
y_train = train_df['Personality']
X_test = test_df[features]

# ロジスティック回帰モデルを定義して学習
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# テストデータで予測を実行
predictions = model.predict(X_test)
print("モデルの学習と予測が完了しました。")


# --- ステップ4: 提出用のファイルを作成 ---
print("\n--- 提出ファイルを作成 ---")
# 予測結果 (0, 1) を元の 'Extrovert'/'Introvert' に戻す
reverse_personality_map = {0: 'Extrovert', 1: 'Introvert'}
prediction_labels = [reverse_personality_map[p] for p in predictions]

# 読み込んでおいた sample_submission_df の 'Personality'列を、我々の予測結果で上書き
submission_df = sample_submission_df.copy() # 元のDFを汚さないようにコピー
submission_df['Personality'] = prediction_labels

# 提出ファイルとして保存 (Kaggleでは /kaggle/working/ 以下に保存されます)
submission_df.to_csv('submission.csv', index=False)

print("\n--- 全ての処理が完了しました！ ---")
print("提出ファイル 'submission.csv' を作成しました。")
print("ファイルの内容 (最初の5行):")
print(submission_df.head())



import pandas as pd
import numpy as np
import os
import lightgbm as lgb

# --- ステップ0: ファイルパスの自動検出とデータの読み込み ---
# Kaggleの/kaggle/input/ディレクトリを探索して、必要なファイルのフルパスを自動的に見つけます。
train_path = ''
test_path = ''
submission_path = ''

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'train.csv' in filename:
            train_path = os.path.join(dirname, filename)
        elif 'test.csv' in filename:
            test_path = os.path.join(dirname, filename)
        elif 'sample_submission.csv' in filename:
            submission_path = os.path.join(dirname, filename)

# 見つけたパスを使って、3つのファイルを読み込みます
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(submission_path)

print("データの読み込みが完了しました。")
print(f"訓練データパス: {train_path}")
print(f"テストデータパス: {test_path}")
print(f"提出ファイルパス: {submission_path}")


# --- ステップ1: 欠損値の処理 ---
print("\n--- 欠損値の処理を開始 ---")
# 数値とカテゴリの列を特定 (idとターゲットは除外)
numeric_cols = train_df.select_dtypes(include=np.number).columns.drop(['id'])
categorical_cols = train_df.select_dtypes(include='object').columns.drop(['Personality'])

# 訓練データから補完値を計算（数値は中央値、カテゴリは最頻値）
imputation_values = {}
for col in numeric_cols:
    imputation_values[col] = train_df[col].median()
for col in categorical_cols:
    imputation_values[col] = train_df[col].mode()[0]

# 訓練データとテストデータの欠損値を補完
for col, value in imputation_values.items():
    train_df[col].fillna(value, inplace=True)
    test_df[col].fillna(value, inplace=True)
print("欠損値の処理が完了しました。")


# --- ステップ2: カテゴリカル変数のエンコーディング ---
print("\n--- エンコーディングを開始 ---")
# マッピング辞書を作成
binary_map = {'No': 0, 'Yes': 1}
personality_map = {'Extrovert': 0, 'Introvert': 1}

# 訓練データとテストデータのカテゴリカル変数を数値に変換
train_df['Stage_fear'] = train_df['Stage_fear'].map(binary_map)
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map(binary_map)
test_df['Stage_fear'] = test_df['Stage_fear'].map(binary_map)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(binary_map)

# 訓練データのターゲット変数(Personality)を数値に変換
train_df['Personality'] = train_df['Personality'].map(personality_map)
print("エンコーディングが完了しました。")


# --- ステップ3: モデルの学習と予測 ---
print("\n--- モデルの学習と予測を開始 ---")
# 訓練データから特徴量 (X) とターゲット (y) を分離
features = [col for col in train_df.columns if col not in ['id', 'Personality']]
X_train = train_df[features]
y_train = train_df['Personality']
X_test = test_df[features]

# LightGBMモデルを定義して学習
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# テストデータで予測を実行
predictions = model.predict(X_test)
print("モデルの学習と予測が完了しました。")


# --- ステップ4: 提出用のファイルを作成 ---
print("\n--- 提出ファイルを作成 ---")
# 予測結果 (0, 1) を元の 'Extrovert'/'Introvert' に戻す
reverse_personality_map = {0: 'Extrovert', 1: 'Introvert'}
prediction_labels = [reverse_personality_map[p] for p in predictions]

# 読み込んでおいた sample_submission_df の 'Personality'列を、我々の予測結果で上書き
submission_df = sample_submission_df.copy()
submission_df['Personality'] = prediction_labels

# ★★★ 提出ファイル名を 'submission.csv' に修正 ★★★
submission_df.to_csv('submission.csv', index=False)

print("\n--- 全ての処理が完了しました！ ---")
print("提出ファイル 'submission.csv' を作成しました。")
print("ファイルの内容 (最後の5行):")
print(submission_df.head())





