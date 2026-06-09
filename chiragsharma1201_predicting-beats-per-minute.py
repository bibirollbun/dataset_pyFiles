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


d=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


d.head()


d.info()


d.describe()


d.isnull().sum()


d.shape


d.dtypes


d.columns


f= [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]


len(f)


d1=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


d1.head()


d1.isnull().sum()


d1.info()


d1.dtypes


d1.columns


d1.describe()


d1.shape


for i in f:
    d[i].fillna(d[i].median())
    d1[i].fillna(d[i].median())


from sklearn.preprocessing import StandardScaler


s=StandardScaler()
d[f]=s.fit_transform(d[f])
d1[f]=s.transform(d1[f])


d[f]


d1[f]


X=d[f]
y=d['BeatsPerMinute']
X_val=d1[f]


from sklearn.model_selection import train_test_split,GridSearchCV


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


X_train.shape


X_test.shape


import xgboost as xg


param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


x=xg.XGBRegressor(
    random_state=42,
    eval_metric='rmse'
)


g= GridSearchCV(
    estimator=x,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)


g.fit(X_train,y_train)


b= g.best_estimator_


g.best_params_


b.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)


from sklearn.metrics import mean_squared_error


y_pred=b.predict(X_test)


np.sqrt(mean_squared_error(y_test,y_pred))


t=b.predict(X_val)


t


submission = pd.DataFrame({
    'id': d1['id'],
    'BeatsPerMinute':t
})


submission.to_csv('/kaggle/working/submission.csv', index=False)


submission.shape


submission.head()


submission.info()

