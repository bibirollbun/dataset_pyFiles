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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# データの読み込み
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# 目的変数（ターゲット）は 'efs'
target = 'efs'

# 学習用データから目的変数と不要な列（例：efs_time）を除外
X = train.drop(columns=[target, 'efs_time'])
y = train[target]

# テストデータは submission 用にIDは保持するので、まずIDを別変数に保存
if 'ID' in test.columns:
    test_ID = test['ID']
else:
    test_ID = np.arange(len(test))
    
# テストデータから不要な列を除外（ターゲット列が含まれている場合など）
X_test = test.copy()
if 'efs' in X_test.columns:
    X_test = X_test.drop(columns=['efs'])
if 'efs_time' in X_test.columns:
    X_test = X_test.drop(columns=['efs_time'])
if 'ID' in X_test.columns:
    X_test = X_test.drop(columns=['ID'])

# 数値型とカテゴリ型の特徴量を識別
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 数値型の特徴量の中にIDがあれば除外（モデル学習には不要）
if 'ID' in numeric_features:
    numeric_features.remove('ID')
    X = X.drop(columns=['ID'])

# 数値特徴量の前処理：欠損値は中央値で補完し、標準化する
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# カテゴリ特徴量の前処理：欠損値は 'missing' で補完し、ワンホットエンコーディングする
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 数値とカテゴリの前処理を ColumnTransformer で統合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 前処理とモデル（ここではランダムフォレスト回帰）を結合したパイプラインを定義
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 学習データを訓練用と検証用に分割（例：8:2）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの学習
model.fit(X_train, y_train)

# 検証データで予測し、RMSE を評価
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse:.4f}")

# テストデータで予測
test_preds = model.predict(X_test)

# 提出用ファイルの作成（ID と予測値）
submission = pd.DataFrame({
    "ID": test_ID,
    "prediction": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


