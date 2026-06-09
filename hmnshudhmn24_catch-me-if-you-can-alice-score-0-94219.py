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


import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]
train = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv', parse_dates = times, index_col='session_id')
test = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv', parse_dates = times, index_col='session_id')
train.sort_values(by='time1', inplace=True)
idx = train.shape[0]
data = pd.concat([train, test], sort=False)
# train.shape, test.shape, data.shape
data[sites] = data[sites].fillna(0).astype(np.uint16) 
data['words'] = data[sites].astype(str).apply(' '.join, axis=1)
words = TfidfVectorizer(max_features=50000, ngram_range=(1, 3)).fit_transform(data['words'])
data.drop(['words'], inplace=True, axis=1)
# words
#make main model 
model = LogisticRegression(random_state=17, solver='liblinear')
time_split = TimeSeriesSplit(n_splits=10)
train.time1.min(), train.time1.max(), test.time1.min(), test.time1.max()


X_train = words[:idx]
y_train = train.target

cv_scores = cross_val_score(model, X_train, y_train, cv=time_split, scoring='roc_auc')
cv_scores, cv_scores.mean()


#host features and time
data['min'] = data[times].min(axis=1)
data['max'] = data[times].max(axis=1)
data['seconds'] = ((data['max'] - data['min']) / np.timedelta64(1, 's'))
data['minutes'] = ((data['max'] - data['min']) / np.timedelta64(1, 'm')).round(2)
data.drop(['min','max'], inplace=True, axis=1)
data['month'] = data['time1'].apply(lambda ts: ts.month+(12*(ts.year-2013))).astype(np.int8)
data['yyyymm'] = data['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
data['mm'] = data['time1'].apply(lambda ts: ts.month).astype(np.int8)
data['yyyy'] = data['time1'].apply(lambda ts: ts.year).astype(np.int8)
data['dayofweek'] = data['time1'].apply(lambda ts: ts.dayofweek).astype(np.int8)
data['weekend'] = data['time1'].apply(lambda ts: ts.dayofweek > 5).astype(np.int8)
data['hour'] = data['time1'].apply(lambda ts: ts.hour).astype(np.int8)
hosts = pd.read_pickle('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl')
hosts = pd.DataFrame(data=list(hosts.keys()), index=list(hosts.values()), columns=['name']) 
hosts['split'] = hosts['name'].str.split('.')
hosts['len'] = hosts['split'].map(lambda x: len(x)).astype(np.int8)
hosts['domain'] = hosts['split'].map(lambda x: x[-1])
hosts.drop(['name','split'], inplace=True, axis=1)
hosts.index.rename('site1', inplace=True)
data = pd.merge(data, hosts, how='left', on='site1')
data.columns


data['short'] = data['minutes'].map(lambda x: x < 0.8).astype(np.int8)
data['long'] = data['minutes'].map(lambda x: x >= 0.8).astype(np.int8)
data["online_day"] = data['time1'].apply(lambda ts: ts.dayofweek in [0,1,3,4]).astype(np.int8)
data["mon"] = data['time1'].apply(lambda ts: ts.dayofweek in [0]).astype(np.int8) # monday
data["wen"] = data['time1'].apply(lambda ts: ts.dayofweek in [2]).astype(np.int8) # wensday
data["sun"] = data['time1'].apply(lambda ts: ts.dayofweek in [6]).astype(np.int8) # sunday
agg = data[data.target==1].groupby(['mm']).seconds.agg({ 'mean', 'sum', 'count'})
agg


data['morning'] = data['time1'].apply(lambda ts: (ts.hour >= 7) & (ts.hour < 12)).astype(np.int8)
data['day'] = data['time1'].apply(lambda ts: (ts.hour >= 12) & (ts.hour < 18)).astype(np.int8)
data['evening'] = data['time1'].apply(lambda ts: (ts.hour >= 18) & (ts.hour < 23)).astype(np.int8)
data['night'] = data['time1'].apply(lambda ts: (ts.hour >= 23) | (ts.hour < 7)).astype(np.int8) 
data['big_site'] = data['len'].apply(lambda x: x > 5).astype(np.int8)
data['typical_site'] = data['len'].apply(lambda x: x == 3).astype(np.int8)
data['typical_domain'] = data['domain'].map(lambda x: x in ('com', 'fr', 'net', 'uk', 'org', 'tv')).astype(int)
data.drop(times + sites + ['target'], inplace=True, axis=1)
data.to_pickle('dump.pkl')
data.columns


data = pd.read_pickle('dump.pkl')
data.drop([
    'seconds', 
    'minutes', 
    'month', 
    'mm', 
    'yyyy', 
    'dayofweek',
    'weekend', 
    'hour', 
    'len', 
    'domain', 
    'short', 
    'long',
    'online_day',
    'mon',
    'wen',
    'sun',
    'big_site',
    'typical_site',
    'typical_domain',
], inplace=True, axis=1)

data = pd.get_dummies(data, columns=[

])

features_to_scale = [
    'yyyymm',
]
data[features_to_scale] = StandardScaler().fit_transform(data[features_to_scale])
X_train = csr_matrix(hstack([words[:idx], data[:idx]]))
y_train = train.target

params = {
    'C': np.logspace(-2, 2, 10),
    'penalty': ['l1','l2']
}

grid = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=time_split, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

grid.best_estimator_, grid.best_score_, grid.best_params_


#Submission
model = grid.best_estimator_
model.fit(X_train, y_train)

X_test = csr_matrix(hstack([words[idx:], data[idx:]]))
y_test = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"session_id": test.index, "target": y_test})
submission.to_csv('submission.csv', index=False)

