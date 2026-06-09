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


# ---------------------------
# 1. ライブラリのインポート
# ---------------------------
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------
# 2. データの読み込み
# ---------------------------
df_train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s4e9/sample_submission.csv")

# ---------------------------
# 3. 特徴量の前処理関数（修正版）
# ---------------------------
def preprocess_modified(df):
    df = df.copy()

    # エンジン気筒数の抽出
    def extract_cylinders(engine_str):
        if isinstance(engine_str, str):
            match = re.search(r'(\d+)\s*Cyl', engine_str)
            if match:
                return int(match.group(1))
        return None

    df['cylinders'] = df['engine'].apply(extract_cylinders)

    # 燃料タイプの簡略分類
    df['fuel_category'] = df['fuel_type'].fillna('Unknown').apply(
        lambda x: 'Gasoline' if 'Gasoline' in x else
                  'Hybrid' if 'Hybrid' in x else
                  'Diesel' if 'Diesel' in x else
                  'Electric' if 'Electric' in x else
                  'Other'
    )

    # 車齢の計算（2024年基準）
    df['car_age'] = 2024 - df['model_year']

    # 欠損処理と不要列の削除
    df['accident'] = df['accident'].fillna('Unknown')
    df['clean_title'] = df['clean_title'].fillna('Unknown')
    df.drop(columns=['engine', 'fuel_type', 'model_year'], inplace=True)

    return df

# ---------------------------
# 4. 前処理の適用
# ---------------------------
train_mod = preprocess_modified(df_train)
test_mod = preprocess_modified(df_test)

# ---------------------------
# 5. 特徴量と目的変数の分離
# ---------------------------
X_train = train_mod.drop(columns=['id', 'price'])
y_train = train_mod['price']
X_test = test_mod.drop(columns=['id'])
test_ids = df_test['id']

# ---------------------------
# 6. 数値／カテゴリ特徴量の特定
# ---------------------------
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numerical_cols = X_train.select_dtypes(include='number').columns.tolist()

# ---------------------------
# 7. 前処理パイプラインの作成
# ---------------------------
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# ---------------------------
# 8. モデルパイプライン（ランダムフォレスト）
# ---------------------------
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# ---------------------------
# 9. モデル学習と評価
# ---------------------------
model_pipeline.fit(X_train, y_train)
train_preds = model_pipeline.predict(X_train)

print(f"R² score: {r2_score(y_train, train_preds):.4f}")
print(f"MAE: {mean_absolute_error(y_train, train_preds):.2f}")
print(f"MSE: {mean_squared_error(y_train, train_preds):.2f}")

# ---------------------------
# 10. テストデータで予測
# ---------------------------
test_preds = model_pipeline.predict(X_test)

# ---------------------------
# 11. 提出ファイルの作成
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'price': test_preds
})
submission.to_csv("submission_modified_features.csv", index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# データの読み込み
df_train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")  # アップロードした train.csv のパス

# 排気量の抽出
def extract_engine_displacement(engine_str):
    if isinstance(engine_str, str):
        match = re.search(r'(\d\.\d)L', engine_str)
        if match:
            return float(match.group(1))
    return None

df_train['engine_displacement'] = df_train['engine'].apply(extract_engine_displacement)

# 相関係数の計算
corr_value = df_train[['engine_displacement', 'price']].corr().iloc[0, 1]

# 相関図（回帰直線付き散布図）
plt.figure(figsize=(10, 6))
sns.regplot(data=df_train, x='engine_displacement', y='price', scatter_kws={'alpha': 0.5})
plt.title(f'Engine Displacement vs. Price (Correlation = {corr_value:.2f})')
plt.xlabel('Engine Displacement (L)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

