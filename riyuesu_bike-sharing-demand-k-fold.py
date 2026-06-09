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
from sklearn.model_selection import KFold
import os

dfs = {}
BASE_PATH = '/kaggle/input/bike-sharing-demand/' 

# データの読み込みと結合
for name in ['train', 'test']:
    file_path = os.path.join(BASE_PATH, f'{name}.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: 指定されたパス '{file_path}' でファイルを読み込めませんでした。")
        raise
    df['_data'] = name
    dfs[name] = df
df_all = pd.concat([dfs['train'], dfs['test']], ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['datetime'])
print("✅ データの読み込みと結合完了")

# 特徴量エンジニアリング
df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month
df_all['hour'] = df_all['datetime'].dt.hour
df_all['weekday'] = df_all['datetime'].dt.weekday 
df_all['day_hour'] = df_all['weekday'].astype(str) + '_' + df_all['hour'].astype(str)
df_all['is_rush_hour'] = df_all['hour'].isin([7, 8, 9, 17, 18]).astype(int)
df_all['is_night'] = df_all['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df_all['temp_humidity_interaction'] = df_all['temp'] * df_all['humidity']
df_all = df_all.drop(['atemp', 'datetime'], axis=1)

categorical_features = ['year', 'month', 'hour', 'weekday', 'season', 'weather', 'holiday', 'workingday', 'day_hour', 'is_rush_hour', 'is_night']
for col in categorical_features:
    df_all[col] = df_all[col].astype('category')
df_all = pd.get_dummies(df_all, columns=categorical_features, drop_first=True)

df_targets = df_all[['count', 'casual', 'registered']].copy()
df_data_flag = df_all['_data'].copy()
df_all = df_all.drop(['count', 'casual', 'registered', '_data'], axis=1)

X_train = df_all[df_data_flag == 'train']
X_test = df_all[df_data_flag == 'test']
y_train_casual_log = np.log1p(df_targets['casual'][df_data_flag == 'train'])
y_train_registered_log = np.log1p(df_targets['registered'][df_data_flag == 'train'])

print("✅ データ準備完了。チューニングを開始します。")

# ================================================================
# K-Fold & ハイパーパラメータチューニング
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# パラメータ
# Registeredモデル
LGBM_PARAMS_REG = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,           # Early Stoppingで調整
    'learning_rate': 0.02,          # 低速学習
    'num_leaves': 35,               # 比較的浅い木
    'min_child_samples': 25,        # ノイズ抑制
    'lambda_l1': 0.1,               # L1正則化を適用
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1
}

# Casualモデル
LGBM_PARAMS_CASUAL = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.05,          # 標準学習率
    'num_leaves': 65,               # 比較的深い木
    'min_child_samples': 5,         # 学習を優先
    'lambda_l2': 0.01,              # L2正則化を弱めに適用
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1
}

test_preds_casual_kf = np.zeros(X_test.shape[0])
test_preds_registered_kf = np.zeros(X_test.shape[0])


for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"--- Fold {fold+1}/{NFOLDS} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    
    # Casual
    y_train_cas, y_val_cas = y_train_casual_log.iloc[train_index], y_train_casual_log.iloc[val_index]
    model_cas = lgb.LGBMRegressor(**LGBM_PARAMS_CASUAL)
    model_cas.fit(
        X_train_fold, y_train_cas,
        eval_set=[(X_val_fold, y_val_cas)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)] # Early Stopping
    )
    test_preds_casual_kf += model_cas.predict(X_test) / NFOLDS
    
    # Registered 
    y_train_reg, y_val_reg = y_train_registered_log.iloc[train_index], y_train_registered_log.iloc[val_index]
    model_reg = lgb.LGBMRegressor(**LGBM_PARAMS_REG)
    model_reg.fit(
        X_train_fold, y_train_reg,
        eval_set=[(X_val_fold, y_val_reg)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    test_preds_registered_kf += model_reg.predict(X_test) / NFOLDS

# 逆変換と合計 
predictions_casual = np.expm1(test_preds_casual_kf)
predictions_registered = np.expm1(test_preds_registered_kf)

predictions_casual = np.maximum(0, predictions_casual)
predictions_registered = np.maximum(0, predictions_registered)

predictions_count = (predictions_casual + predictions_registered).astype(int)

# ファイルの作成
submission = pd.DataFrame({
    'datetime': dfs['test']['datetime'],
    'count': predictions_count
})
submission.to_csv('submission_tuned_kf.csv', index=False)

print("\n✅ チューニング済み K-Fold 分割モデルの予測完了。")
print("ファイル名: submission_kf.csv として保存されました。")


