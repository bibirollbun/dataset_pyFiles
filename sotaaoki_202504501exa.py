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


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# === 1. データ読み込み & ラベルエンコーディング ===
train = pd.read_csv("../input/playground-series-s4e4/train.csv")
test  = pd.read_csv("../input/playground-series-s4e4/test.csv")
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex']  = le.transform(test['Sex'])


# 特徴量と目的変数の分割
drop_cols = ['id', 'Rings']
X_train = train.drop(columns=drop_cols)
y_train = train['Rings']
X_test  = test.drop(columns=['id'])

# === (A) 目的変数に1を足して log1p 変換 ===
y_train = np.log1p(y_train)


# === 2. 特徴量エンジニアリング ===
class AbaloneFeatureEngineer(TransformerMixin, BaseEstimator):
    """
    多様な特徴候補を一括生成:
      - 比率特徴, 対数変換, 多項式, 交互作用
    """
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # 変数名更新：データフレームの列に合わせる
        sw  = X['Shell weight']        # 殻重量
        di  = X['Diameter']            # 直径
        hw  = X['Whole weight']        # 全体重量
        shw = X['Whole weight.1']      # 身体重量 (Shucked weight)
        vw  = X['Whole weight.2']      # 内臓重量 (Viscera weight)
        # Ratio features
        X['shell_whole_ratio']   = sw / hw
        X['shucked_shell_ratio'] = shw / sw
        X['viscera_shell_ratio'] = vw / sw
        # Log transform
        X['log_ShellWeight']     = np.log1p(sw)
        X['log_Diameter']        = np.log1p(di)
        X['log_WholeWeight']     = np.log1p(hw)
        X['log_ShuckedWeight']   = np.log1p(shw)
        X['log_VisceraWeight']   = np.log1p(vw)
        # Polynomial features
        X['sw_sq']               = sw ** 2
        X['di_sq']               = di ** 2
        X['hw_sq']               = hw ** 2
        X['shw_sq']              = shw ** 2
        X['vw_sq']               = vw ** 2
        # Interaction features
        X['sw_di_mul']           = sw * di
        X['sw_hw_mul']           = sw * hw
        X['sw_shw_mul']          = sw * shw
        X['sw_vw_mul']           = sw * vw
        X['di_hw_mul']           = di * hw
        X['di_shw_mul']          = di * shw
        X['di_vw_mul']           = di * vw
        return X


# === 3. 自作モデルのアンサンブルによる学習 ===
estimators = [
    ('xgb',   xgb.XGBRegressor(random_state=71, verbosity=0)),
    ('rf',    RandomForestRegressor(random_state=0)),
    ('ridge', Ridge(random_state=0))
]
pipeline = Pipeline([
    ('feat_eng', AbaloneFeatureEngineer()),
    ('scale',    StandardScaler()),
    ('ensemble', VotingRegressor(estimators))
])


# === 4. モデル学習 ===
pipeline.fit(X_train, y_train)

# === 5. 予測 & 逆変換 ===
preds_log = pipeline.predict(X_test)
preds = np.expm1(preds_log)


# === 6. 提出ファイル作成 ===
submission = pd.DataFrame({'id': test['id'], 'Rings': preds})
submission.to_csv('submission_ensemble_log.csv', index=False)
print("✅ Ensemble model with log-target transform: submission_ensemble_log.csv")

















# sfm = SelectFromModel(pipeline.named_steps['model'], threshold='median', prefit=True)
# selected = feat_imp[feat_imp >= feat_imp.median()].index.tolist()
# print("Selected features based on median importance:", selected)

# X_train_sel = X_train_fe[selected]
# X_test_fe   = pipeline.named_steps['feat_eng'].transform(X_test)
# X_test_sel  = X_test_fe[selected]

# model_sel = xgb.XGBRegressor(random_state=71, verbosity=0)
# model_sel.fit(X_train_sel, y_train)





