import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col = 'id')
sam_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


# orig_1 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
# orig_2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
# orig_3 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


# train = pd.concat([train, orig_1, orig_2, orig_3], axis = 0)


train.head(2)


train.shape


import seaborn as sns
import matplotlib.pyplot as plt


n = len(train.select_dtypes(exclude = object).columns)
mask = np.triu(np.ones((n, n), dtype=int))


plt.figure(figsize = (15, 12));
sns.heatmap(train.select_dtypes(exclude = object).corr(), mask = mask, annot = True);
# plt.yticks([]);


train['night'] = (train['lighting']=='night').astype(int)
train['no_clear'] = (train['weather']!='clear').astype(int)
train['speed_60'] = (train["speed_limit"] >= 60).astype(int)
train['accidents_2'] = (np.array(train["num_reported_accidents"]) > 2).astype(int)
train['num_accidents&curvature']  = train['num_reported_accidents'] * train['curvature']
train['speed_limit&curvature']  = train['speed_limit'] * train['curvature']
train['num_lanes'] = train['num_lanes']**8


test['night'] = (test['lighting']=='night').astype(int)
test['no_clear'] = (test['weather']!='clear').astype(int)
test['speed_60'] = (test["speed_limit"] >= 60).astype(int)
test['accidents_2'] = (np.array(test["num_reported_accidents"]) > 2).astype(int)
test['num_accidents&curvature']  = test['num_reported_accidents'] * test['curvature']
test['speed_limit&curvature']  = test['speed_limit'] * test['curvature']
test['num_lanes'] = test['num_lanes']**8


def rmse(y, ypred):
    return np.sqrt(np.sum((y-ypred)**2)/len(y))


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


feats = list(train.select_dtypes(exclude = object).columns)
feats.remove('accident_risk')


X = train[feats]  # Features
y = train['accident_risk'] 


test_predict = test[feats]


from sklearn.model_selection import KFold


fin_pred = np.zeros(test.shape[0])
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index] 
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ypred = np.round(np.clip(y_pred, 0, 1), 2)
    print(rmse(y_test, ypred))
    y_pred = model.predict(test_predict)
    ypred = np.round(np.clip(y_pred, 0, 1), 2)
    fin_pred += ypred/num_folds


y_pred = model.predict(X_test)
ypred = np.round(np.clip(y_pred, 0, 1), 2)


rmse(y_test, ypred)


sam_sub.head()


sam_sub.accident_risk = fin_pred
LR_pred = fin_pred


sam_sub.to_csv('submission_LR.csv', index = False)


CATS = train.select_dtypes(object).columns


X = train.drop('accident_risk', axis = 1)
y = train.accident_risk


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=420)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))


X.select_dtypes(object).columns


import warnings
warnings.simplefilter('ignore')
TARGET = 'accident_risk'


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'---Fold {fold+1}/{N_SPLITS}---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

    # X_test = test[FEATURES].copy()
    X_test = test.copy()

    X_train[CATS] = X_train[CATS].astype('category')    
    X_val[CATS] = X_val[CATS].astype('category')    
    X_test[CATS] = X_test[CATS].astype('category')    
    
    model = XGBRegressor(
        n_estimators=100000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        device='cuda',
        early_stopping_rounds=200,
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=500, 
             )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] += val_preds

    test_preds += model.predict(X_test)

    print(f"Fold {fold+1} RMSE: {mean_squared_error(y_val, val_preds, squared=False)}")
    print(f"Fold {fold+1} R2: {r2_score(y_val, val_preds)}")

test_preds /= N_SPLITS

print(f"Overall OOF RMSE: {mean_squared_error(y, oof_preds, squared=False):.5f}")
print(f"Overall OOF R2: {r2_score(y, oof_preds):.5f}")


len(model.feature_importances_)


cols = list(train.columns)


cols.remove('accident_risk')


import seaborn as sns
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'feature': cols, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 20))
sns.barplot(x='importance', 
            y='feature', 
            data=importance_df.head(50)) 
plt.title('Feature Importance (Fold5 model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


res = pd.DataFrame({'id': test.index, TARGET: np.round(test_preds, 2)})
XGB_pred = test_preds


res.to_csv('submission_XGB.csv', index=False)


from lightgbm import LGBMRegressor, log_evaluation, early_stopping


cat_cols = list(train.select_dtypes(object).columns)


lgbm_params = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "colsample_bytree": 0.43,
    "learning_rate": 0.016,
    "max_depth": 12,
    "min_child_samples": 67,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 50,
    "random_state": 42,
    "verbose": -1,
    'categorical_feature' : cat_cols,
    "callbacks": [
        log_evaluation(period=100), 
        early_stopping(stopping_rounds=100)
    ]
}


lgbm_model = LGBMRegressor(**lgbm_params)


train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')


X_train, X_test, y_train, y_test = train_test_split(train.drop('accident_risk', axis = 1), train['accident_risk'], test_size = 0.25)


lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
    eval_metric='rmse')


preds = lgbm_model.predict(X_test)
y_pred = np.clip(preds, 0, 1)


rmse(y_test, y_pred)


y_pred = lgbm_model.predict(test)
ypred = np.round(np.clip(y_pred, 0, 1), 2)


res = pd.DataFrame({'id': test.index, TARGET: np.round(ypred, 2)})
LGBM_pred = ypred


res.to_csv('submission_LGBM.csv', index=False)


preds = (LR_pred + XGB_pred + LGBM_pred)/3
res = pd.DataFrame({'id': test.index, TARGET: np.round(preds, 2)})
res.to_csv('submission_ENS.csv', index=False)

