import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
%matplotlib inline


traindata = pd.read_csv("/kaggle/input/kobe-bryant-shot-selection/data.csv.zip")
traindata.head()


nona = traindata[traindata['shot_made_flag'].notna()]


alpha = 0.02
plt.figure(figsize=(10,8))

plt.subplot(121)
plt.scatter(nona['loc_x'], nona['loc_y'], color='blue', alpha=alpha)
plt.title('loc_x and loc_y')

plt.subplot(122)
plt.scatter(nona['lon'], nona['lat'], color='green', alpha=alpha)
plt.title('lat and lon')


traindata['dist'] = np.sqrt(traindata['loc_x']**2 + traindata['loc_y']**2)
traindata['angle'] = np.arctan2(traindata['loc_y'], traindata['loc_x'])
traindata[['dist', 'angle']].head()


traindata['remaining_time'] = traindata['minutes_remaining']*60 + traindata['seconds_remaining']
traindata['remaining_time'].head()


traindata['season'].unique()


traindata['season'] = traindata['season'].str.split('-').str.get(1)
traindata['season'].unique()


drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    traindata = traindata.drop(drop, axis=1)
traindata.head(10)


categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    traindata = pd.concat([traindata, pd.get_dummies(traindata[var], prefix=var)], axis=1)
    traindata = traindata.drop(var, axis=1)
traindata.tail(10)


nona = traindata[traindata['shot_made_flag'].notna()]
submission = traindata[traindata['shot_made_flag'].isnull()]
submission = submission.drop('shot_made_flag', axis=1)
submission.head()


data = nona.drop('shot_made_flag', axis=1)
target = nona['shot_made_flag']


from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target)


rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(Xtrain, Ytrain)
score = rfc.score(Xtest, Ytest)
print("Random Forest:{}".format(score))


def logloss(act, pred):
    epsilon = 1e-15
    pred = np.maximum(epsilon, pred)       
    pred = np.minimum(1-epsilon, pred)     
    ll = np.sum(act * np.log(pred) + (1 - act) * np.log(1 - pred))  
    ll = ll * -1.0 / len(act)
    return ll


import time
from sklearn.model_selection import KFold

print('Finding best n_estimators for RandomForestClassifier...')

min_score = 1000000
best_n = 0
scores_n = []
range_n = [1, 10, 25, 50, 75, 100]

for n in range_n:
    print("the number of tress: {}".format(n))
    
    rfc_score = 0
    rfc = RandomForestClassifier(n_estimators=n)
    
    kf = KFold(n_splits=10, shuffle=True)
    
    t1 = time.time()
    for train_index, test_index in kf.split(data):
        rfc.fit(data.iloc[train_index], target.iloc[train_index])
        pred_proba = rfc.predict_proba(data.iloc[test_index])[:, 1]
        rfc_score += logloss(target.iloc[test_index], pred_proba) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
print(best_n, min_score)


print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
range_m = [1, 10, 25, 50, 75, 100]
for m in range_m:
    print("the max depth : {0}".format(m))
    
    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    
    kf = KFold(n_splits=10, shuffle=True)
    
    t1 = time.time()
    for train_index, test_index in kf.split(data):
        rfc.fit(data.iloc[train_index], target.iloc[train_index])
        pred_proba = rfc.predict_proba(data.iloc[test_index])[:, 1]
        rfc_score += logloss(target.iloc[test_index], pred_proba) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m
    
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
print(best_m, min_score)


model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(data, target)
pred = model.predict_proba(submission)


output = pd.read_csv("/kaggle/input/kobe-bryant-shot-selection/sample_submission.csv.zip")
output['shot_made_flag'] = pred
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

