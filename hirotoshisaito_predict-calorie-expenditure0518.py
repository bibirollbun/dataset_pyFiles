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




# 必要なライブラリをインポート
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# データの読み込み（必要に応じてパスを修正）
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# ---------------------------
# 性別（Sex）のエンコード
# ---------------------------
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# ---------------------------
# log変換（目的変数）
# ---------------------------
train['Calories_log'] = np.log1p(train['Calories'])

# ---------------------------
# 特徴量とターゲットの分離（全特徴量使用）
# ---------------------------
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X = train[features]
y = train['Calories_log']
X_test = test[features]

# ---------------------------
# スケーリング
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# XGBoost モデルの定義と学習
# ---------------------------
xgb_model = XGBRegressor(
    objective='reg:squarederror',  # log変換後の連続値に対して
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_scaled, y)

# ---------------------------
# テストデータに対する予測（log → exp変換）
# ---------------------------
y_pred_log = xgb_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

# ---------------------------
# 提出ファイルの作成
# ---------------------------
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_pred
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

# ---------------------------
# RMSLE 評価（学習データで）
# ---------------------------
y_pred_train_log = xgb_model.predict(X_scaled)
y_pred_train = np.expm1(y_pred_train_log)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_pred_train))
print(f'RMSLE: {rmsle:.5f}')

# ---------------------------
# 提出ファイルの上位50件を表示
# ---------------------------
print("\n=== 提出ファイル（上位50行） ===")
print(submission.head(50))

