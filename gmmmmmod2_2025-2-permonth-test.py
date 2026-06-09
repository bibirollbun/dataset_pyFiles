import numpy as np 
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import optuna
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def missing(df):
	missing_number = df.isnull().sum().sort_values(ascending=False)
	missing_percent = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
	missing_values = pd.concat([missing_number, missing_percent], axis=1)
	return missing_values

def NullValueProcess(df, cat_cols,num_cols):
    df = df.copy()
    
    for col in cat_cols:
        df.fillna({col: 'None'}, inplace=True)
        df[col] = df[col].astype('category')
        
    for col in num_cols:
        df.fillna({col: -1}, inplace=True)
    return df


def CVtest(X,Y,model):
    
    scores = []
    cv = KFold(shuffle=True,n_splits=5)

    for train_idx, test_idx in cv.split(X,Y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_test)
            
        score = mean_squared_error(Y_test, Y_pred)
        score = np.sqrt(score)
            
        scores.append(score)

    # print(f'CVtest score is {np.mean(scores)}')
    return np.mean(scores)


def CV_Objective(trial,X,Y):
    params = {
            'objective': trial.suggest_categorical('objective', ['poisson', 'tweedie', 'regression']),
            'random_state': trial.suggest_int('random_state',0,100000),
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
        }
    if params['objective'] == 'tweedie':
        params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)

    LGBM = LGBMRegressor(**params)
    
    score = CVtest(X,Y,LGBM)
    return score

def RunOputna(X,Y):
    study = optuna.create_study(direction='minimize')# 初始化参数学习器，并指定方向 'direction'
    
    study.optimize(lambda trial: CV_Objective(trial, X, Y), n_trials=10)
     
    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value}")
    return study.best_params


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
val = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv',index_col='id')
samples_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv',index_col='id')

cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 
            'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']
train = NullValueProcess(train,cat_cols,num_cols)
test = NullValueProcess(test,cat_cols,num_cols)
val = NullValueProcess(val,cat_cols,num_cols)


model = LGBMRegressor()


X = train.drop(columns=['Price'])
Y = train['Price']

# params = RunOputna(X,Y)


params = {
    'objective': 'regression', 
    'random_state': 65754, 
    'n_estimators': 249, 
    'max_depth': 3, 
    'learning_rate': 0.03234251060806044, 
    'subsample': 0.6432231416987411, 
    'colsample_bytree': 0.588805980813428, 
    'min_data_in_leaf': 85
}
model = LGBMRegressor(**params)


X_val = val.drop(columns=['Price'])
Y_val = val['Price']

model.fit(X,Y)
res = model.predict(X_val)

score = mean_squared_error(Y_val, res)
np.sqrt(score)


samples_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
model.fit(X,Y)

ans = model.predict(test)

submission = pd.DataFrame({
     "id" : samples_df['id'],
     "Price": ans
})
    
submission.to_csv(f"submission.csv",index=False)


submission.head()

