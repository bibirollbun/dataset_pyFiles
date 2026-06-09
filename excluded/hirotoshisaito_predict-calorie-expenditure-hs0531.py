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


# データ読み込み
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# 確認
print(train.shape)
print(test.shape)
train.head()



test.head()


# カラムの情報確認
train.info()

# 統計量確認
train.describe()

# カテゴリ変数の確認
print(train['Sex'].value_counts())

# 欠損値の有無
print(train.isnull().sum())



## ターゲット（Calories）の分布確認

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train['Calories'], bins=20, kde=True)
plt.title('Calories Distribution')
plt.show()



# 特徴量との相関係数（数値）

# 相関係数（ターゲットとの）
corr = train.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
print(corr)




# ライブラリのインポート
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

# データ読み込み
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ログ変換した目的変数の作成
train['Calories_log'] = np.log1p(train['Calories'])

# BMI計算
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)

# 年齢カテゴリの関数
def age_group(age):
    if age < 20:
        return 'teen'
    elif age < 35:
        return 'young_adult'
    elif age < 50:
        return 'middle_age'
    else:
        return 'senior'

# 年齢をカテゴリに変換
train['AgeGroup'] = train['Age'].apply(age_group)
test['AgeGroup'] = test['Age'].apply(age_group)

# 元のAge列を削除
train.drop(columns=['Age'], inplace=True)
test.drop(columns=['Age'], inplace=True)

# One-hotエンコーディング
train = pd.get_dummies(train, columns=['Sex', 'AgeGroup'])
test = pd.get_dummies(test, columns=['Sex', 'AgeGroup'])

# trainとtestの列を揃える（id列は除外）
train, test = train.align(test, join='left', axis=1, fill_value=0)

# 相互作用特徴量の追加
train['BMI_Duration'] = train['BMI'] * train['Duration']
test['BMI_Duration'] = test['BMI'] * test['Duration']

train['BMI_HeartRate'] = train['BMI'] * train['Heart_Rate']
test['BMI_HeartRate'] = test['BMI'] * test['Heart_Rate']

# 特徴量と目的変数の分離
y = train['Calories_log']
X = train.drop(columns=['id', 'Calories', 'Calories_log'])  # 不要列を除外
X_test = test.drop(columns=['id'])  # テストデータのid列も除外

# 列の順序を統一（明示的に）
X_test = X_test[X.columns]

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 学習/検証データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# モデル構築（EarlyStopping付き）
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=30,
    verbose=True
)

# 予測と逆変換
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

# 提出用ファイル作成
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_pred
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

# 学習データでRMSLE評価（任意）
y_train_pred_log = model.predict(X_scaled)
y_train_pred = np.expm1(y_train_pred_log)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_train_pred))
print(f'RMSLE: {rmsle:.5f}')


