import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


train_df.duplicated().sum()


train_df = train_df.drop_duplicates()
train_df.duplicated().sum()


train_df.shape, test_df.shape, sample_sub.shape


train_df.info()


test_df.info()


X = train_df.copy().drop(columns='accident_risk')
y = train_df['accident_risk']
test = test_df.copy()


def create_optimized_features(df):
    """
    Создание оптимизированных признаков на основе корреляционного анализа
    """
    df = df.copy()

    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_curvature'] = np.log1p(df['curvature'])

    return df


X = create_optimized_features(X)
test = create_optimized_features(test)


numeric_features = X.select_dtypes(include=['number']).columns
object_features = X.select_dtypes(include=['object', 'bool']).columns
numeric_features, object_features


scaler = RobustScaler()
encoder = LabelEncoder()


for num_col in numeric_features:
    scaler.fit_transform(X[[num_col]]).flatten()
    X[num_col] = scaler.transform(X[[num_col]])
    test[num_col] = scaler.transform(test[[num_col]])

for obj_col in object_features:
    encoder.fit_transform(X[obj_col]).flatten()
    X[obj_col] = encoder.transform(X[obj_col])
    test[obj_col] = encoder.transform(test[obj_col])


X


# params from Optuna
LGB_best_params = {
    'n_estimators': 3000,
    'learning_rate': 0.014291084943047132,
    'num_leaves': 200,
    'max_depth': 20,
    'min_child_samples': 11,
    'min_child_weight': 0.002,
    'subsample': 0.7378723574719579,
    'subsample_freq': 1,
    'colsample_bytree': 0.9223783646573634,
    'reg_alpha': 8.003237221691328e-08,
    'reg_lambda': 1.9726751536952425e-05,
    'min_split_gain':  0.004,
    'feature_fraction': 0.9 ,
}


X_reset = X.reset_index(drop=True)
y_reset = y.reset_index(drop=True)


LGB_model = LGBMRegressor(**LGB_best_params,
                          objective='regression',
                          metric='rmse',
                          boosting_type='gbdt',
                          random_state=42,
                          n_jobs=-1,
                          verbose=-1)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train LightGBM
    LGB_model.fit(X_tr, y_tr)
    y_pred = LGB_model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold}: {rmse:.5f}")

simple_avg_score = np.mean(cv_scores)
print(f"\nSimple Average CV Score: {simple_avg_score:.5f} (+/- {np.std(cv_scores):.5f})")


LGB_model.fit(X_reset, y_reset)

sub_pred = LGB_model.predict(test)

sample_sub['accident_risk'] = sub_pred
sample_sub


sample_sub.to_csv("submission.csv", index=False)
print("submission.csv готов!")

