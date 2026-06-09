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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_submit = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


df_train.head()


df_test.head()


df_train.info()


df_train.describe()


df_test.info()


df_train['Sex'].value_counts()



from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'female' else 0)


df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'female' else 0)


df_train.head()


X, y = df_train.drop(['Calories', 'id'], axis=1), df_train['Calories']
y_log = np.log1p(y)


X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)


from xgboost import XGBRegressor

# モデルの初期化（基本設定）
model = XGBRegressor(
    objective='reg:squarederror',  # RMSEを最小化する回帰目的
    n_estimators=100,              # 最大の木の数（ラウンド数）
    learning_rate=0.1,             # 学習率（小さくすると精度UP、収束遅い）
    max_depth=3,                   # 各決定木の深さ
    random_state=42,               # 再現性確保
    early_stopping_rounds=10       # 検証スコアが向上しなければ10ラウンドで打ち切り
)


model.fit(
    X_train, y_train_log,
    eval_set=[(X_valid, y_valid_log)],  # 検証セットを渡す
    verbose=True                    # ログ出力（Falseにすると非表示）
)


from sklearn.metrics import mean_squared_log_error, r2_score

# 検証データで予測
y_pred_log = model.predict(X_valid)
y_pred = np.expm1(y_pred_log)
y_valid = np.expm1(y_valid_log)

# 評価指標
rmsle = mean_squared_log_error(y_valid, y_pred, squared=False)
r2 = r2_score(y_valid, y_pred)

print(f"RMSLE: {rmsle:.4f}")
print(f"R2 Score: {r2:.4f}")


import matplotlib.pyplot as plt
from xgboost import plot_importance

# グラフで可視化（デフォルトは weight）
plot_importance(model)
plt.show()


# 1. 説明変数だけ取り出す
X_test = df_test.drop('id', axis=1)

# 2. 予測
y_pred_test_log = model.predict(X_test)
y_pred_test = np.expm1(y_pred_test_log)

# 3. DataFrameで結合
submission = df_test[['id']].copy()
submission['Calories'] = y_pred_test
submission.to_csv('submission.csv', index=False)


y_pred_test.min()


sub = pd.read_csv('/kaggle/working/submission.csv')
sub.head()




