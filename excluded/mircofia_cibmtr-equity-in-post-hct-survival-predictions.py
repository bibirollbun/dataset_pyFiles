!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import lightgbm as lgb
import seaborn as sns
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train_data.head()


table = PrettyTable()
target = 'efs'
notsel = 'efs_time'
table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train_data.columns:
    data_type = str(train_data[column].dtype)
    non_null_count_train= np.round(100-train_data[column].count()/train_data.shape[0]*100,1)
    if column!=target and column!=notsel:
        non_null_count_test = np.round(100-test_data[column].count()/test_data.shape[0]*100,1)
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


float64_columns = train_data.select_dtypes(include=['float64']).columns
print(float64_columns.tolist())


float64_columns = train_data.select_dtypes(include=['float64']).columns
float64_columns_with_missing = train_data[float64_columns].columns[train_data[float64_columns].isna().any()].tolist()
print(float64_columns_with_missing)


columns_to_fill = [
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6',
    'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high',
    'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low',
    'hla_match_dqb1_low', 'hla_match_a_high', 'donor_age',
    'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high',
    'comorbidity_score', 'karnofsky_score', 'hla_low_res_8',
    'hla_match_drb1_high', 'hla_low_res_10'
]


def knn_imputation(train_data, columns_to_fill):
    imputer = KNNImputer(n_neighbors=5)
    train_data[columns_to_fill] = imputer.fit_transform(train_data[columns_to_fill])
    return train_data
columns_to_fill = ['hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 'donor_age', 'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 'comorbidity_score', 'karnofsky_score', 'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10']
train_data = knn_imputation(train_data, columns_to_fill)
test_data = knn_imputation(test_data, columns_to_fill)


def check_missing_values(train_data, columns_to_fill):
    missing_counts = train_data[columns_to_fill].isnull().sum()
    print(missing_counts)
columns_to_fill = ['hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low', 'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 'donor_age', 'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 'comorbidity_score', 'karnofsky_score', 'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10']
check_missing_values(train_data, columns_to_fill)


object_columns = train_data.select_dtypes(include=['object']).columns
train_data[object_columns] = train_data[object_columns].fillna('unknown')


object_columns1 = test_data.select_dtypes(include=['object']).columns
test_data[object_columns1] = test_data[object_columns1].fillna('unknown')


# !pip install lifelines


from lifelines import KaplanMeierFitter
def transform_survival_probability_kmf(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train_data["KaplanMeier"] = transform_survival_probability_kmf(train_data, time_col='efs_time', event_col='efs')


plt.figure(figsize=(10, 6))
sns.kdeplot(train_data.loc[train_data.efs == 1, "KaplanMeier"], label="efs=1, Yes Event", shade=True)
sns.kdeplot(train_data.loc[train_data.efs == 0, "KaplanMeier"], label="efs=0, No Event", shade=True)
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time")
plt.legend()
plt.show()


from sklearn.preprocessing import LabelEncoder
object_columns = train_data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in object_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])


object_columns1 = test_data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in object_columns1:
    test_data[col] = label_encoder.fit_transform(test_data[col])


# train_data.to_csv('train_data.csv', index=False)
# test_data.to_csv('test_data.csv', index=False)


# train_data1 = pd.read_csv('/kaggle/input/cibmtr-equity-in-post-hct-survival-data/train_data.csv')
# test_data1 = pd.read_csv('/kaggle/input/cibmtr-equity-in-post-hct-survival-data/test_data.csv')


X = train_data.drop(columns=['KaplanMeier','efs','efs_time'])
y = train_data['KaplanMeier']


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_val_pred = xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("(MSE):", mse)
print("R^2 :", r2)


from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(xgb_model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
mse_scores = -mse_scores  
print("KFold  MSE :", mse_scores)
print("KFold  MSE :", mse_scores.mean())

r2_scores = cross_val_score(xgb_model, X_train, y_train, cv=kfold, scoring='r2')
print("KFold  R^2 :", r2_scores)
print("KFold  R^2:", r2_scores.mean())


import optuna
import logging
optuna.logging.set_verbosity(logging.CRITICAL)
def objective(trial):
    params1 = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    xgb_model = XGBRegressor(random_state=42, **params1,tree_method='gpu_hist', gpu_id=0)
    xgb_model.fit(X_train, y_train)
    y_val_pred = xgb_model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    return mse
study1 = optuna.create_study(direction='minimize')
study1.optimize(objective, n_trials=100,n_jobs=-1)

print("best: ", study1.best_params)


best_xgb_model = XGBRegressor(random_state=42, **study1.best_params)
best_xgb_model.fit(X_train, y_train)
y_val_pred = best_xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("MSE: ", mse)
print("R^2 : ", r2)


from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor(random_state=42,verbose=0) 
catboost_model.fit(X_train, y_train)
y_val_pred = catboost_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("MSE:", mse)
print(" R^2 :", r2)


from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)  
lgbm_model.fit(X_train, y_train)
y_val_pred = lgbm_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("MSE:", mse)
print("R^2:", r2)


def objective(trial):
    params3 = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42,
        'verbose': -1
    }
    lgbm_model = LGBMRegressor(**params3,device='gpu')
    lgbm_model.fit(X_train, y_train)
    y_val_pred = lgbm_model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    return mse
study3 = optuna.create_study(direction='minimize')
study3.optimize(objective, n_trials=100,n_jobs=-1)
print("best: ", study3.best_params)



best_lgbm_model = LGBMRegressor(**study3.best_params)
best_lgbm_model.fit(X_train, y_train)
y_val_pred = best_lgbm_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print("MSE: ", mse)
print("R2 : ", r2)


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
base_models = [
    ('xgb', XGBRegressor(tree_method='gpu_hist', gpu_id=0,**study1.best_params)),
    ('catboost', CatBoostRegressor(task_type='GPU', devices='0',verbose=0)),
    ('lgbm', LGBMRegressor(device='gpu',**study3.best_params))
]
meta_model = LinearRegression()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)
y_val_pred_stacking = stacking_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred_stacking)
r2 = r2_score(y_val, y_val_pred_stacking)
print("MSE:", mse)
print("R^2 :", r2)


y_val_pred_stacking1 = stacking_model.predict(test_data)


su = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


su['prediction'] = y_val_pred_stacking1


su


su.to_csv('submission.csv', index=False)




