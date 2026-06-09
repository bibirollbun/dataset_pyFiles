# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

train_df.drop("id", axis=1, inplace=True)

train_df.head()


def feature_engineering(df):
    df['Loudness_for_Energy']=df['AudioLoudness']*df['Energy']
    df['Vocal_Acoustic_Instrumental']=df['VocalContent']*df['AcousticQuality']*df['InstrumentalScore']
    df['Rhythm_Mood']=df['RhythmScore']*df['MoodScore']
    df['Live_Duration']=df['TrackDurationMs']/(df['LivePerformanceLikelihood']+1e-6)

    return df

train_df=feature_engineering(train_df)
test_df=feature_engineering(test_df)


train_df.count()


train_df.isnull().sum().sort_values(ascending=True)


train_df.info()


train_df.describe()


for col in train_df.columns:
    print(f"{train_df[col].describe()}")
    # print("Skew")
    print(f"Skew: {train_df[col].skew()} \n")


# # train_df['InstrumentalScore'] = np.log1p(train_df['InstrumentalScore'])

# from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler #MinMax since every data in graph is between 0 and 1 while not the two I scaled using StandardScaler so to manage that we do MinMax

# scaler=RobustScaler()
# train_df = scaler.fit_transform(train_df.columns)
# # train_df['AudioLoudness']=scaler.fit_transform(train_df[['AudioLoudness']])


def managing_skew(df, ref_df=None):
    outlier_cols=['AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
                 'Loudness_for_Energy','Vocal_Acoustic_Instrumental', 'Rhythm_Mood', 'Live_Duration']

    if ref_df is None:
        ref_df=df
        
    for col in outlier_cols:
        low,high=ref_df[col].quantile([0.005,0.995])
        df[col]=df[col].clip(lower=low, upper=high)

    posit_skew=['VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood','Vocal_Acoustic_Instrumental', 'Rhythm_Mood', 'Live_Duration']
    for col in posit_skew:
        df[col]=np.log1p(df[col])

    df['AudioLoudness']=np.cbrt(df['AudioLoudness'])
    df['Loudness_for_Energy']=np.cbrt(df['Loudness_for_Energy'])
    return df

train_sk_df=managing_skew(train_df)
test_sk_df=managing_skew(test_df, ref_df=train_sk_df)


for col in train_df.columns:
    if col=="BeatsPerMinute":
        continue
    plt.figure(figsize=(4,4))
    train_df[col].plot(kind='hist', label=col, bins=15, alpha=0.25)
    plt.legend()
    
plt.show()


for col in train_df.columns:
    if col=="BeatsPerMinute":
        continue
    plt.figure(figsize=(4,4))
    train_df[col].plot(kind='kde', label=col)
    plt.legend()

plt.show()


from sklearn.model_selection import train_test_split

X=train_df.drop("BeatsPerMinute", axis=1)
y=train_df['BeatsPerMinute']

X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.2, random_state=42)


# Trying LightGBM
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna

def objective(trial):
    params={
        'objective':'regression_l1',
        'metric':'rmse',
        'n_estimators':100000,
        'boosting_type':'gbdt',
        'n_jobs':-1,
        'device':'gpu',
        'gpu_platform_id':0,
        'gpu_device_id':0,

        'learning_rate':trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth':trial.suggest_int('max_depth', 3,8),
        'num_leaves':trial.suggest_int('num_leaves', 20,300),
        'reg_alpha':trial.suggest_float('reg_aplha', 1e-8, 10, log=True),
        'reg_gamma':trial.suggest_float('reg_gamma', 1e-8, 10, log=True),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample':trial.suggest_float('subsample', 0.4, 1.0),
    }

    model=lgb.LGBMRegressor(
        **params
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])
    
    y_preds=model.predict(X_val)
    RMSE=np.sqrt(mean_squared_error(y_val, y_preds))
    # print(f"Validation RMSE: {RMSE
    return RMSE

study=optuna.create_study(
    direction='minimize',
    study_name='lgbm_regression_tuning'
)

study.optimize(
    objective,
    n_trials=100,
    show_progress_bar=True,
)

print("\n--- OPTIMIZATION FINISHED ---")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best Validation RMSE: {study.best_value}")
print(f"Best hyperparameters: ")

for key, value in study.best_params.items():
    print(f"{key}:{value}")



best_params = study.best_params

best_params['n_estimators'] = 10000 
best_params['random_state'] = 42
best_params['boosting_type'] = 'gbdt'
best_params['device'] = 'gpu'
best_params['gpu_platform_id'] = 0
best_params['gpu_device_id'] = 0
best_params['n_jobs'] = -1

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(X_val,y_val)

print("Final model trained successfully on all available training data.")


test_df.head()


predictions = final_model.predict(test_df.drop(['id'], axis=1))

# 2. Create the submission DataFrame
# This matches the structure in your image
submission_df = pd.DataFrame({
    'Id': test_df['id'],
    'BeatsPerMinute': predictions
})

# 3. Save the file in the required format
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' has been saved!")
print("\nSubmission Preview:")
print(submission_df.head())




