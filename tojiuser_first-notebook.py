import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
print(train.head(2))


train['num_sold'] = train['num_sold'].fillna(train['num_sold'].mean())
train["num_sold"] = np.log1p(train["num_sold"])
X = train.drop(columns=['num_sold', 'id']).astype('str').astype("category")
y = train['num_sold']

x_test = test.drop(columns=['id']).astype('str').astype("category")
train_pool = Pool(data=X, label=y, cat_features=X.columns.values)
test_pool = Pool(data=x_test, cat_features=x_test.columns.values)


cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=X.columns.values,
    loss_function='RMSE',
    verbose=100
)


cat_model.fit(train_pool)


cat_train = cat_model.predict(train_pool).reshape(-1,1)
cat_test = cat_model.predict(test_pool).reshape(-1,1)


import xgboost as xgb

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train["num_sold"] = np.log1p(train["num_sold"])

train_X = train.drop(columns=['id','num_sold']).astype('category')
train_y = train['num_sold'].fillna(train['num_sold'].mean())
test_X = test.drop(columns='id').astype('category')


dtrain = xgb.DMatrix(train_X, label=train_y, enable_categorical=True)
dtest = xgb.DMatrix(test_X, enable_categorical=True)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42,
}
xgb_model = xgb.train(params, dtrain, 1000, verbose_eval=100)
xgb_train = xgb_model.predict(dtrain).reshape(-1,1)
xgb_test = xgb_model.predict(dtest).reshape(-1,1)


import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1)
lgb_model.fit(X, y)
lgb_train = lgb_model.predict(X).reshape(-1,1)
lgb_test = lgb_model.predict(x_test).reshape(-1,1)

stacked_train = np.hstack((xgb_train, cat_train, lgb_train))
stacked_test = np.hstack((xgb_test, cat_test, lgb_test))


from sklearn.linear_model import Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(stacked_train, train_y)


y_pred = np.expm1(meta_model.predict(stacked_test))


Test = pd.DataFrame({
    'id': test['id'],
    'num_sold': y_pred
})


Test.head()

