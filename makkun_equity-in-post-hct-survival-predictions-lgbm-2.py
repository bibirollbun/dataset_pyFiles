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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# データの読み込み
train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")  # 訓練データ
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")    # 予測対象データ



# 目的変数と説明変数の設定
X = train_df.drop(columns=['efs_time', 'efs', 'ID'])  # ID, efs_time, efsは削除
y = train_df['efs']  # 目的変数（1: イベント発生, 0: 発生なし）



# テストデータの説明変数
X_test = test_df.drop(columns=['ID'])  # IDのみ削除（efs, efs_timeは元々ない）



# 数値カラムとカテゴリカラムの分離
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()



# 前処理パイプライン
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])



# 訓練データと検証データの分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# LightGBMデータセット作成
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)
X_test_transformed = preprocessor.transform(X_test)

train_data = lgb.Dataset(X_train_transformed, label=y_train)
valid_data = lgb.Dataset(X_valid_transformed, label=y_valid)



# LightGBM パラメータ設定
params = {
    'objective': 'binary',  # バイナリ分類
    'metric': 'auc',  # AUCスコアを最適化
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'verbose': -1
}


# モデル学習
#model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=500, early_stopping_rounds=50)
# モデル学習（early_stopping_roundsの設定）
model = lgb.train(params, train_data, valid_sets=[valid_data], 
                  valid_names=["valid"], num_boost_round=500)



# 予測（リスクスコア = イベント発生確率）
risk_scores = model.predict(X_test_transformed)



# 提出用ファイルの作成
submission = pd.DataFrame({'ID': test_df['ID'], 'prediction': risk_scores})
submission.to_csv('/kaggle/working/submission.csv', index=False)



