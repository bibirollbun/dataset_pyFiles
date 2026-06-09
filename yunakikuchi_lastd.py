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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation


# データ読み込み
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# 特徴量生成
def create_features(df):
    df['soil_crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    df['npk_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['npk_ratio'] = df['npk_total'] / (df['Moisture'] + 1)
    df['temp_hum_ratio'] = df['Temparature'] / (df['Humidity'] + 1)
    df['is_high_nitro'] = (df['Nitrogen'] > 60).astype(int)
    
    for col in ['Soil Type', 'Crop Type', 'soil_crop']:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

train = create_features(train)
test = create_features(test)


# ターゲットエンコード
le = LabelEncoder()
train['target'] = le.fit_transform(train['Fertilizer Name'])


# 特徴量一覧
features = [col for col in train.columns if col not in ['id', 'Fertilizer Name', 'target']]


# モデル設定
N_FOLDS = 3 
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train), len(le.classes_)))
test_preds = np.zeros((len(test), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    print(f'Fold {fold + 1}')
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = X_train['target'], X_val['target']

    lgb_train = lgb.Dataset(X_train[features], label=y_train)
    lgb_val = lgb.Dataset(X_val[features], label=y_val)

    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'learning_rate': 0.05,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'seed': 42,
    }

    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=300,  
        callbacks=[
            early_stopping(stopping_rounds=25),  
            log_evaluation(100)
        ]
    )

    oof_preds[val_idx] = model_lgb.predict(X_val[features])
    test_preds += model_lgb.predict(test[features]) / N_FOLDS


top_3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
submission = pd.DataFrame()
submission['id'] = test['id']
submission['Fertilizer Name'] = [' '.join(le.inverse_transform(row)) for row in top_3]
submission.to_csv('submission.csv', index=False)

