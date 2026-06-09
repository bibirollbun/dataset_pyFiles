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
import re
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df_sub=pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv", nrows=10000)
df_test=pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv", nrows=10000)



def preprocess_data(df):
    df = df.copy()

    # milage を float に変換
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
    df['milage'] = df['milage'].fillna(df['milage'].mean())

    # car_age を生成
    current_year = 2025
    if 'model_year' in df.columns:
        df['car_age'] = current_year - df['model_year']

    # 補完処理（存在する列のみ）
    for col in ['clean_title', 'int_color', 'ext_color']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # 不要な列を削除（存在する場合のみ）
    drop_cols = ['id', 'model_year', 'engine', 'fuel_type']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df


# --- 両方のデータセットに前処理を適用 ---
print("データの前処理を開始します...")
train_df_processed = preprocess_data(df_train)
test_df_processed = preprocess_data(df_test)
print("データの前処理が完了しました。")

# 特徴量 (X) とターゲット (y) を分離
X_train = train_df_processed.drop('price', axis=1)
y_train = train_df_processed['price']
X_test = test_df_processed.copy()

# 'id'列は予測には使用しないため削除
if 'id' in X_train.columns:
    X_train = X_train.drop('id', axis=1)

if 'id' in X_test.columns:
    X_test = X_test.drop('id', axis=1)
# --- カテゴリ変数と数値変数の特定 ---
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("カテゴリカル特徴量:", categorical_features)
print("数値特徴量:", numerical_features)


# 両方のデータセットに前処理を適用
train_df_processed = preprocess_data(df_train)
test_df_processed = preprocess_data(df_test)


# 特徴量 (X) とターゲット (y) を分離
X_train = train_df_processed.drop('price', axis=1)
y_train = train_df_processed['price']


# テストデータのIDを保持
test_ids = df_test['id']
X_test = test_df_processed.copy()


# 訓練データとテストデータで列の整合性を確認
# OneHotEncoderのhandle_unknown='ignore'により、テストデータに訓練データにないカテゴリが出ても問題ないが、
# モデル列の多すぎるユニーク値は精度に影響するため、上位N件でまとめるなどの工夫も考えられます。
# 例として、brandやmodelなどはユニーク値が多い可能性があります。

# カテゴリカル特徴量と数値特徴量を特定
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=[np.number]).columns
# 数値特徴量とカテゴリカル特徴量のための前処理パイプラインを作成
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')) # 欠損した数値データを平均値で補完
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 欠損したカテゴリカルデータを最頻値で補完
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # カテゴリカル特徴量をOne-Hotエンコーディング
])
# 異なる列に異なる変換を適用するためのColumnTransformerを作成
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # その他の列はそのまま通過させる（例: 'id'列など）
)


 #前処理とRandomForestRegressorモデルを含む完全なパイプラインを作成
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]) # n_jobs=-1で全コア利用


# モデルを訓練
print("モデルの学習を開始します...")
model.fit(X_train, y_train)
print("モデルの学習が完了しました。")
# テストデータで予測を実行
print("テストデータで予測を実行します...")
test_predictions = model.predict(X_test)
print("予測が完了しました。")


# 訓練データでの予測
train_predictions = model.predict(X_train)

# R^2スコアの計算
r2_train = r2_score(y_train, train_predictions)
print(f"訓練データのR^2 Score: {r2_train:.4f}")

# オプション: MAEやMSEも計算できます
mae_train = mean_absolute_error(y_train, train_predictions)
print(f"訓練データのMAE: {mae_train:.4f}")

mse_train = mean_squared_error(y_train, train_predictions)
print(f"訓練データのMSE: {mse_train:.4f}")


# 提出用のDataFrameを作成
# df_subをベースにIDと予測値をマージする
submission_df = pd.DataFrame({'id': test_ids, 'price': test_predictions})
# 価格が非負であることを保証
submission_df['price'] = submission_df['price'].apply(lambda x: max(0, x))
# 提出ファイルを保存 (Kaggleの出力ディレクトリに保存することが一般的)
submission_df.to_csv('submission.csv', index=False)


print("予測結果はsubmission.csvに保存されました。")
print("サンプル提出ファイルの形式:")
print(df_sub.head())
print("\n生成された提出ファイルの形式:")
print(submission_df.head())

