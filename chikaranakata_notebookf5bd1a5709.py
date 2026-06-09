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

# train.csv は ./data/train.csv にある
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    print("train.csv の読み込みに成功しました！")
    print("train_df の形状:", train_df.shape) #train_dfが何行何列かを表示（95, 2）
    print("\ntrain_df の最初の5行:")
    print(train_df.head())
except FileNotFoundError: #もしエラーが見つからなかったらエラーを表示出る
    print("エラー: ./data/train.csv が見つかりません。")

# test.csv は、なかった
try:
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("\ntest.csv の読み込みに成功しました！")
    print("test_df の形状:", test_df.shape)
    print("\ntest_df の最初の5行:")
    print(test_df.head())
except FileNotFoundError:
    print("エラー: ./data/test.csv が見つかりません。")


# カラムをデータ型ごとに分類して表示

int_cols = train_df.select_dtypes(include=['int']).columns.tolist()
float_cols = train_df.select_dtypes(include=['float']).columns.tolist()
object_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print("--- 整数型のカラム ---")
print(int_cols)

print("\n--- 浮動小数点型のカラム ---")
print(float_cols)

print("\n--- オブジェクト型（文字列など）のカラム ---")
print(object_cols)

# 他のデータ型があれば、必要に応じて追加
# datetime_cols = train_df.select_dtypes(include=['datetime']).columns.tolist()
# print("\n--- 日時型のカラム ---")
# print(datetime_cols)


from sklearn.preprocessing import LabelEncoder

# LabelEncoderを準備
le = LabelEncoder()

# train_dfの'Stage_fear'カラムを直接変換し、上書きする
train_df['Stage_fear'] = le.fit_transform(train_df['Stage_fear'])

# train_dfの中身を確認
print("--- 変換後のDataFrame ---")
display(train_df.head())
print("\n'Stage_fear'が数値に変わっていることを確認:")
print(train_df.info())


# 'Drained_after_socializing'カラムの欠損値を先に埋める（LabelEncoderは欠損値があるとエラーになるため）
train_df['Drained_after_socializing'].fillna('Unknown', inplace=True)

# LabelEncoderを準備
le_drained = LabelEncoder()

# train_dfの'Drained_after_socializing'カラムを数値に変換
train_df['Drained_after_socializing'] = le_drained.fit_transform(train_df['Drained_after_socializing'])

print("--- 'Drained_after_socializing' 変換後のデータ ---")
display(train_df.head())


# Personality専用のLabelEncoderを準備
le_personality = LabelEncoder()

# trainデータの'Personality'カラムを数値に変換
# .fit_transform()で変換ルール学習と適用を一度に行います
train_df['Personality'] = le_personality.fit_transform(train_df['Personality'])

# 変換ルールを確認
print("--- 'Personality'の変換ルール ---")
for i, class_name in enumerate(le_personality.classes_):
    print(f"'{class_name}'  ->  {i}")


print("\n--- 全ての変換が完了したデータ ---")
display(train_df.head())


# --- test_dfの前処理 ---

# 'Stage_fear'の変換（学習で使ったleを再利用）
test_df['Stage_fear'] = le.transform(test_df['Stage_fear'])

# 'Drained_after_socializing'の欠損値処理と変換（学習で使ったle_drainedを再利用）
test_df['Drained_after_socializing'].fillna('Unknown', inplace=True)
test_df['Drained_after_socializing'] = le_drained.transform(test_df['Drained_after_socializing'])


print("--- 前処理後のテストデータ ---")
display(test_df.head())


print("--- 変換前のテストデータ ---")
print(test_df.info())


# 変換したいカラムのリスト
columns_to_encode = ['Stage_fear', 'Drained_after_socializing']

for col in columns_to_encode:
    # LabelEncoderを準備
    le = LabelEncoder()

    # 1. 学習データだけで変換ルールを学習させる
    le.fit(train_df[col])

    # 2. 学習したルールを使ってテストデータを変換する
    test_df[col] = le.transform(test_df[col])


print("\n--- 変換後のテストデータ ---")
print(test_df.info())

print("\n--- 変換後のデータ（先頭5行）---")
display(test_df.head())



# 各カラムの欠損値の数を合計して表示
print(train_df.isnull().sum())


# 学習データの中央値を計算
medians = train_df.median()

# 学習データとテストデータの欠損値を、学習データの中央値で埋める
train_df.fillna(medians, inplace=True)
test_df.fillna(medians, inplace=True)

# 処理が成功したか、もう一度欠損値の数を確認
print("--- 処理後の欠損値の数 (学習データ) ---")
print(train_df.isnull().sum())

print("\n--- 処理後の欠損値の数 (テストデータ) ---")
print(test_df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns # ← この行を追加します

# グラフを見やすいようにサイズを調整
plt.figure(figsize=(10, 8))

# 相関行列を計算
correlation_matrix = train_df.corr()

# ヒートマップを作成
# annot=Trueで数値を表示、cmap='coolwarm'で色分け、fmt='.2f'で小数点以下2桁まで表示
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('heatmap')
plt.show()


import lightgbm as lgb
# le_personality（'Personality'を変換したLabelEncoder）が
# 事前に作成されている必要があります。

# --- 1. データを学習用と予測用に準備 ---
# 特徴量X と 目的変数y
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']

# 予測するためのテストデータ
X_test = test_df.drop('id', axis=1)


# --- 2. モデルを学習させる ---
model = lgb.LGBMClassifier(random_state=42)
model.fit(X, y)


# --- 3. テストデータで予測を行う ---
# モデルは数値(0, 1)で予測結果を出す
predictions_encoded = model.predict(X_test)


# --- 4. 予測結果を文字列に戻す ---
# le_personalityを使って数値から'Introvert'/'Extrovert'に変換
predictions_text = le_personality.inverse_transform(predictions_encoded)


# --- 5. 提出用のDataFrameを作成 ---
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions_text
})


# --- 6. CSVファイルとして保存 ---
# index=False を指定しないと、余計な列が追加されるので注意
submission_df.to_csv('submission.csv', index=False)


print("提出ファイル 'submission.csv' を作成しました！")
display(submission_df.head())

