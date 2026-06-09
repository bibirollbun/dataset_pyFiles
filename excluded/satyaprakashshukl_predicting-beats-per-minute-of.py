import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold



df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


df_train.head()


df_train.describe()


df_test.head()


df_sub.head()


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)



df_train.shape,df_test.shape


df_train.isnull().sum()


df_test.isnull().sum()


df_train.shape,df_test.shape,df_sub.shape


df_train.corr()


df_train.dtypes



y = df_train['BeatsPerMinute']
X = df_train.drop(columns=['BeatsPerMinute'])


X_test = df_test


X



df_sub.head()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



rf_params_gpu = {
    'n_estimators': 100,
    'max_depth': 12,
    'max_features': 'sqrt',
    'min_samples_split': 6,
    'min_samples_leaf': 2,
    'bootstrap': True,
    'random_state': 42
   
}
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
"""X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.28, random_state=42)
model = RandomForestRegressor(**rf_params)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
score = rmse(y_val, val_pred)
print(f"Validation RMSE: {score:.4f}")
test_preds = model.predict(X_test)"""


import lightgbm as lgb

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 25,
    'learning_rate': 0.005,
    'n_estimators': 650,
    'max_depth': 32,
    'min_child_samples': 15,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'device': 'gpu',          
    'gpu_platform_id': 0,    
    'gpu_device_id': 0       
}

lgbm_params = {
    'subsample': 0.6, 
    'reg_lambda': 0.5, 
    'reg_alpha': 0.1, 
    'num_leaves': 31, 
    'n_estimators': 300, 
    'min_child_samples': 50, 
    'max_depth': 12, 
    'learning_rate': 0.01, 
    'colsample_bytree': 1.0,
    'device': 'gpu',         
    'gpu_platform_id': 0,    
    'gpu_device_id': 0       
}



model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    early_stopping_rounds=100,
    verbose=100
)

val_pred = model.predict(X_val)
score = rmse(y_val, val_pred)
print(f"Validation RMSE: {score:.4f}")

test_predsl = model.predict(X_test)


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

base_estimators = [
    ('lgbm', LGBMRegressor(**lgbm_params,random_state=42)),
    ('rf', RandomForestRegressor(**rf_params_gpu))
]

stack = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=1.0),
    passthrough=False,
    cv=5,                
    n_jobs=-1
)

print('Training>>>>>>>')
stack.fit(X, y)
print('Predicting>>>>>>>')
y_pred = stack.predict(X_test)




y_pred


df_sub['BeatsPerMinute'] = y_pred*0.9+test_predsl*0.1


df_sub.to_csv('submission.csv', index=False)


df_sub.head()


df_sub['BeatsPerMinute'].hist()

