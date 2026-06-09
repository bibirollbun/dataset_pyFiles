import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head(3)


train.dtypes


train.info()


train.isna().sum()


train.describe().T


test.head(2)


test.info()


test.isna().sum()


test.describe().T


X=train.drop(columns=['id','BeatsPerMinute'])
y=train['BeatsPerMinute']
test_id=test['id']
test=test.drop(columns='id',axis=1)


from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 22,
    'learning_rate': 0.005,
    'n_estimators': 475,
    'max_depth': 30,
    'min_child_samples': 15,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}


model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100)]
)

val_pred = model.predict(X_val)
score = rmse(y_val, val_pred)
print(f"Validation RMSE: {score:.4f}")


predictions1 = model.predict(test)


cat_params = {
    'objective': 'RMSE',
    'iterations': 475,
    'learning_rate': 0.005,
    'depth': 6,
    'l2_leaf_reg': 1.0,
    'eval_metric':'RMSE',
    'random_seed': 42,
    'verbose': 0
}


model2 = CatBoostRegressor(**cat_params)

model2.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=100
)

val_pred = model2.predict(X_val)
score = rmse(y_val, val_pred)
print(f"Validation RMSE: {score:.4f}")


predictions2 = model2.predict(test)


predictions = predictions1*0.9 + predictions2*0.1


submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


submission.head(2)




