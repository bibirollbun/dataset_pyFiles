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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.utils.class_weight import compute_sample_weight

# load data & change formats
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
original = original.rename(columns={'pressure ':'pressure','humidity ':'humidity','cloud ':'cloud',
                                    '         winddirection':'winddirection'})
original['rainfall']=original['rainfall'].replace({'yes': 1, 'no': 0})
train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)

# deal with null value
test['winddirection'].fillna(test['winddirection'].mean(),inplace = True)
original['winddirection'].fillna(original['winddirection'].mean(), inplace = True)
original['winddirection'].fillna(original['winddirection'].mean(), inplace = True)
train_real = pd.concat([train,original])
train_real.index = range(train_real.shape[0])

# basic feature engineering
# 1.tempdiff feature
train_real['tempdiff'] = train_real['maxtemp'] - train_real['mintemp']
test['tempdiff'] = test['maxtemp'] - test['mintemp']

# 2.temp humidity interaction feature
train_real['temp_humidity_interaction'] = train_real['temparature'] * train_real['humidity']
test['temp_humidity_interaction'] = test['temparature'] * test['humidity']

# 3.dewpoint + humidity feature
train_real['dewpoint+humidity'] = train_real['dewpoint'] + train_real['humidity']
test['dewpoint+humidity'] = test['dewpoint'] + test['humidity']

# 4.cloud related features
train_real['cloud+humidity'] = train_real['cloud'] + train_real['humidity']
test['cloud+humidity'] = test['cloud'] + test['humidity']
train_real['cloud+humidity+sunshine'] = train_real['cloud'] + train_real['humidity'] + train_real['sunshine']
test['cloud+humidity+sunshine'] = test['cloud'] + test['humidity'] + test['sunshine']

# 5.logged sunshine feature
train_real['sunshine'] = np.log1p(train_real['sunshine'].clip(lower=0))
test['sunshine'] = np.log1p(test['sunshine'].clip(lower=0))

# 6.time shift feature
for c in ['pressure', 'maxtemp', 'temparature', 'humidity', 'mintemp', 'dewpoint', 'cloud', 'sunshine', 'tempdiff']:
    train_real[c+'_shift'] = train_real[c].shift(1)
    test[c+'_shift'] = test[c].shift(1)

# 7.deal with null value again
train_real=train_real.fillna(0)
test=test.fillna(0)

# time & date features
train_real['season']=train_real['day']%365
test['season']=test['day']%365

def get_season(day):
    month=(day%365)//30+1
    if month in [12,1,2]:
        return 0
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    else:
        return 3

train_real['season']=train_real['day'].apply(get_season)
test['season']=test['day'].apply(get_season)

train_real['day_of_year']=train_real['day']%365
test['day_of_year']=test['day']%365


# prepare train data
X = train_real.drop('rainfall', axis=1)
y = train_real['rainfall']

# calculate sample weights
sample_weights = compute_sample_weight('balanced', y)

# standardscaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test)

# define Optuna objective function
def objective(trial):
    C = trial.suggest_float('C', 1e-5, 100, log=True)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    
    # create logistic regression model
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_weights = sample_weights[train_idx]
        
        model.fit(X_train, y_train, sample_weight=train_weights)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        scores.append(score)
    
    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best parameters:', study.best_params)
print('Best cross-validation score:', study.best_value)


# use best params to train final model
best_params = study.best_params
solver = 'liblinear' if best_params['penalty'] == 'l1' else 'lbfgs'

final_model = LogisticRegression(
    C=best_params['C'],
    class_weight=best_params['class_weight'],
    max_iter=best_params['max_iter'],
    penalty=best_params['penalty'],
    solver=solver,
    random_state=42
)

final_model.fit(X_scaled, y, sample_weight=sample_weights)

y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]


test1=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
# generate subbmission file
submission = pd.DataFrame({
    'id': test1['id'],
    'rainfall': y_pred_proba
})

# save submission file
submission.to_csv('submission.csv', index=False)

