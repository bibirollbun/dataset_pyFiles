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


from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import cupy as cp
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,QuantileTransformer
import cupy as cp
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.shape, test.shape


train.head(2)


test.head(2)


qt = QuantileTransformer(output_distribution='normal')


for df in [train, test]:
    df['RhythmScore_sin'] = np.sin(df['RhythmScore'])
    df['RhythmScore_cos'] = np.cos(df['RhythmScore'])
    df['MoodScore_sin'] = np.sin(df['MoodScore'])
    df['MoodScore_cos'] = np.cos(df['MoodScore'])
    df['Energy_cos'] = np.cos(df['Energy'])
    df['Energy_sin'] = np.sin(df['Energy'])    
    df['AudioLoudness_log'] = np.log1p(df['AudioLoudness'])
    df['AudioLoudness_sqrt'] = np.sqrt(df['AudioLoudness'])
    df['VocalContent_log'] = np.log1p(df['VocalContent'])
    df['VocalContent_sqrt'] = np.sqrt(df['VocalContent'])
    df['AcousticQuality_log'] = np.log1p(df['AcousticQuality'])
    df['AcousticQuality_sqrt'] = np.sqrt(df['AcousticQuality'])
    df['InstrumentalScore_log'] = np.log1p(df['InstrumentalScore'])
    df['InstrumentalScore_sqrt'] = np.sqrt(df['InstrumentalScore'])
    df['LivePerformanceLikelihood_sqrt'] = np.sqrt(df['LivePerformanceLikelihood'])
    df['LivePerformanceLikelihood_sqrt'] = np.sqrt(df['LivePerformanceLikelihood'])
    df['Ratio_vocal_content_instumental_score'] = df['VocalContent'] / (df['VocalContent'] + df['InstrumentalScore'])
    df['Ratio_AudioLoudness_instumental_score'] = df['AudioLoudness'] / (df['AudioLoudness'] +  df['InstrumentalScore'])
    df['Ratio_AudioLoudness_AcousticQuality'] = df['AudioLoudness'] / (df['AudioLoudness'] +  df['AcousticQuality'])
    df['Ratio_AudioLoudness_VocalContent'] = df['AudioLoudness'] / (df['AudioLoudness'] +  df['VocalContent'])
    df['Ratio_AudioLoudness_LivePerformanceLikelihood'] = df['AudioLoudness'] / (df['AudioLoudness'] +  df['LivePerformanceLikelihood'])
    df['Ratio_VocalContent_AcousticQuality'] = df['VocalContent'] / (df['VocalContent'] +  df['AcousticQuality'])
    df['Ratio_VocalContent_InstrumentalScore'] = df['VocalContent'] / (df['VocalContent'] +  df['InstrumentalScore'])
    df['AudioLoudness_quantile'] = qt.fit_transform(df[['AudioLoudness']])
    df['VocalContent_quantile'] = qt.fit_transform(df[['VocalContent']])
    df['AcousticQuality_quantile'] = qt.fit_transform(df[['AcousticQuality']])
    df['InstrumentalScore_quantile'] = qt.fit_transform(df[['InstrumentalScore']])
    df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)
    df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
    df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
    df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs'])
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01)  # Avoid division by zero
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  # Scaled for better range
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / (df['TrackDurationMs']/1000)


scaler = StandardScaler()


train_sc = scaler.fit_transform(train[[col for col in train.columns if col not in ['id','BeatsPerMinute']]])
test_sc = scaler.transform(test[[col for col in test.columns if col not in ('id')]])


train_sc.shape, test_sc.shape


X = train_sc
y = train['BeatsPerMinute']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42 )


xgb = XGBRegressor(
    objective =  'reg:squarederror',
    random_state = 42,
    #device = 'cuda',
    n_jobs = -1,
)


param_dist_xgb = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': np.arange(3, 15, 1),
    'learning_rate': np.linspace(0.01, 0.3, 30),
    'subsample': np.linspace(0.5, 1.0, 6),
    'colsample_bytree': np.linspace(0.5, 1.0, 6),
    'gamma': np.linspace(0, 5, 11),
    'min_child_weight': np.arange(1, 10, 1)
}


random_search_xgb = RandomizedSearchCV(
    estimator = xgb,
    param_distributions =  param_dist_xgb,
    n_iter = 30,
    scoring = 'neg_mean_squared_error',
    cv = 3,
    verbose = 0,
    random_state = 42,
    n_jobs =-1
)


random_search_xgb.fit(X_train, y_train)


print(f"Best params: {random_search_xgb.best_params_}")
print(f"Best model: {random_search_xgb.best_estimator_}")


y_pred = random_search_xgb.best_estimator_.predict(X_test)


rmse_xsboost = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"RMSE XGBOOST: {rmse_xsboost}")


test_pred_xgboost = random_search_xgb.best_estimator_.predict(test_sc)


lgbm = LGBMRegressor(
    #device='gpu', 
    random_state=42
)


param_dist_lgbm = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}


random_search_lgbm_regr = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist_lgbm,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)


random_search_lgbm_regr.fit(X_train, y_train)


print(f"Best params: {random_search_lgbm_regr.best_params_}")
print(f"Best model: {random_search_lgbm_regr.best_estimator_}")


y_pred = random_search_lgbm_regr.best_estimator_.predict(X_test)


rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"RMSE LGBM: {rmse_lgbm}")


test_pred_lgbm = random_search_lgbm_regr.best_estimator_.predict(test_sc)


catboost = CatBoostRegressor(
    #task_type='GPU',
    verbose=0,  # Suppress training output
    random_state=42
)


param_dist_catboost = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 300, 500],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0.0, 0.5, 1.0]
}



random_search_catboost = RandomizedSearchCV(
    estimator=catboost,
    param_distributions=param_dist_catboost,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


random_search_catboost.fit(X_train, y_train)


print(f"Best params: {random_search_catboost.best_params_}")
print(f"Best model: {random_search_catboost.best_estimator_}")


y_pred = random_search_catboost.best_estimator_.predict(X_test)


rmse_catboost = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"RMSE Catboost: {rmse_catboost}")


test_pred_catboost = random_search_catboost.best_estimator_.predict(test_sc)


print(f"RMSE XGBOOST: {rmse_xsboost}, RMSE LGBM: {rmse_lgbm}, RMSE Catboost: {rmse_catboost} ")


total_weight = 1 / rmse_xsboost + 1 / rmse_lgbm + 1 / rmse_catboost


xgboost_weight = (1/rmse_xsboost) / total_weight
lgbm_weight = (1/rmse_lgbm) / total_weight
catboost_weight = (1/rmse_catboost) / total_weight


print(f"XGBoost weight: {xgboost_weight} , LGBM weight: {lgbm_weight}, Catboost weight: {catboost_weight}  ")


test_pred_ensemble = test_pred_xgboost * xgboost_weight + test_pred_lgbm * lgbm_weight + test_pred_catboost * catboost_weight


submission = pd.DataFrame(
    {
        'id': test['id'],
        'BeatsPerMinute': test_pred_ensemble
    }
)


assert submission.shape[0] == test.shape[0]


submission.head(2)


submission.to_csv('submission.csv', index=False)
print("Submission created")

