import pandas as pd
from pathlib import Path
import numpy as np


DATAPATH = Path('../input/playground-series-s3e1')
N_ESTIMATORS = 100_000

train = pd.read_csv(DATAPATH/'train.csv')
test = pd.read_csv(DATAPATH/'test.csv')
sample_sub = pd.read_csv(DATAPATH/'sample_submission.csv')


train.head()


train.id.value_counts().max()


train.isna().sum()


train.shape[0], test.shape[0]


from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
target = 'MedHouseVal'


from sklearn.datasets import fetch_california_housing

original_data = fetch_california_housing()
train = train.drop(columns='id')
original_data = pd.DataFrame(data=np.hstack([original_data['data'], original_data['target'].reshape(-1, 1)]), columns=train.columns)

train = pd.concat([train, original_data]).reset_index(drop=True)


clfs = []
rmses = []

params= {
 'lambda_l1': 1.945,
 'num_leaves': 87,
 'feature_fraction': 0.79,
 'bagging_fraction': 0.93,
 'bagging_freq': 4,
 'min_data_in_leaf': 103,
 'max_depth': 17,
}

kf = KFold(n_splits=10, random_state=0, shuffle=True)
for train_index, val_index in kf.split(train):
    X_train, X_val = train[features].loc[train_index], train[features].loc[val_index]
    y_train, y_val = train[target][train_index], train[target][val_index]
    
    clf = LGBMRegressor(learning_rate=0.02, n_estimators=N_ESTIMATORS, metric='rmse', **params)
    clf.fit(X_train.values, y_train, eval_set=[(X_val, y_val)], callbacks=[lgbm.early_stopping(85, verbose=True)])
    preds = clf.predict(X_val.values)
    
    clfs.append(clf)
    rmses.append(mean_squared_error(y_val, preds, squared=False))
print(f'mean RMSE across all folds: {np.mean(rmses)}')


for i in clf.feature_importances_.argsort()[::-1]:
    print(features[i], clf.feature_importances_[i]/clf.feature_importances_.sum())


from catboost import CatBoostRegressor

rmses = []
kf = KFold(n_splits=10, random_state=1, shuffle=True)
for train_index, val_index in kf.split(train):
    X_train, X_val = train[features].loc[train_index], train[features].loc[val_index]
    y_train, y_val = train[target][train_index], train[target][val_index]

    clf = CatBoostRegressor(iterations=N_ESTIMATORS, loss_function='RMSE')
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=1000, verbose=False)
    
    
    preds = clf.predict(X_val.values)
    
    clfs.append(clf)
    rmses.append(mean_squared_error(y_val, preds, squared=False))
print(f'mean RMSE across all folds: {np.mean(rmses)}')


test_preds = []

for clf in clfs:
    preds = clf.predict(test[features].values)
    test_preds.append(preds)


test_preds = np.stack(test_preds).mean(0)
test_preds


submission = pd.DataFrame(data={'id': test.id, 'MedHouseVal': test_preds})
submission.head()


submission.to_csv('submission.csv', index=False)

