# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
import optuna


# df_pipe = df.sample(n = 50000, random_state=42) --> This code was used for trial and tests
df_pipe = df.copy()
X_ = df_pipe.drop(columns = 'BeatsPerMinute')
y_ = df_pipe['BeatsPerMinute']

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.2, random_state=42)

"""
BaseEstimator: Provides get_params and set_params for compatibility with sklearn pipelines and hyperparameter tuning.
TransformerMixin: Provides a default fit_transform method, so custom transformers can be used seamlessly in pipelines.
"""
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TrackDurationMin'] = X['TrackDurationMs']/60000
        X = X.drop(columns=['TrackDurationMs', 'id'])
        X['TotalScore'] = X['RhythmScore'] + X['InstrumentalScore'] + X['MoodScore']
        X['EnergyPerMin'] = X['Energy']/X['TrackDurationMin']
        X['stamina'] = X['Energy']*X['Energy']*X['MoodScore']*X['TrackDurationMin']
        X['resistance'] = (X['Energy']*X['Energy'])/X['MoodScore']
        X['Instrumental_Vocal_Per_Min'] = (X['InstrumentalScore'] + X['VocalContent'])/X['TrackDurationMin']
        X['RhythmPerMin'] = X['RhythmScore']/X['TrackDurationMin']
        X['Instrumental_Beats'] = (X['InstrumentalScore'] - X['VocalContent'])*X['AudioLoudness']
        # print(X)
        return X


"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Standard Scaling and power transformation
"""
pow_cols = ['AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore',
            'LivePerformanceLikelihood', 'EnergyPerMin', 'stamina', 'resistance', 'Instrumental_Vocal_Per_Min',
            'RhythmPerMin']

preprocessor = ColumnTransformer(
    transformers = [
        ('power_t', PowerTransformer(method='yeo-johnson'), pow_cols)
    ],
    remainder = 'passthrough'
)

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Optuna
"""
def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 600, 5000),
        'max_depth' : trial.suggest_int('max_depth', 3, 12),
        'learning_rate' : trial.suggest_float('learning_rate', 0.001, 2, log=True),
        'subsample' : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  #L1
        'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  #L2
    }
    
    """
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Pipeline
    """
    
    pipeline = Pipeline([
        ('features', FeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize")  # maximize R²
study.optimize(objective, n_trials=30)  # try 30 different sets

print("Best score:", study.best_value)
print("Best params:", study.best_params)


import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf
)

# After running study.optimize(...)
plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_parallel_coordinate(study).show()
plot_slice(study).show()
plot_contour(study).show()
plot_edf(study).show()



df_results = study.trials_dataframe()

# Show top rows
df_results.head()


best_params = study.best_params


X = df_pipe.drop(columns = ['BeatsPerMinute'])
y = df_pipe['BeatsPerMinute']

pipe_final = Pipeline([
        ('features', FeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', XGBRegressor(random_state=42, n_jobs=-1, **best_params))
    ])

pipe_final.fit(X, y)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
X_test_final = df_test


y_pred_final = pipe_final.predict(X_test_final)


submission = pd.DataFrame({
    'id' : X_test_final['id'],
    'BeatsPerMinute' : y_pred_final
})


submission = submission.reset_index(drop=True)


submission.to_csv('submission.csv', index=False)

