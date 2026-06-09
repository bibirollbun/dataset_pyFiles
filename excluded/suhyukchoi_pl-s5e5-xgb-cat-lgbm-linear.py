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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


train = pd.read_csv('../input/playground-series-s5e5/train.csv', index_col = 0)
test = pd.read_csv('../input/playground-series-s5e5/test.csv', index_col = 0)
submit = pd.read_csv('../input/playground-series-s5e5/sample_submission.csv')
original_train = pd.read_csv('../input/calories-burnt-prediction/calories.csv', index_col = 0)

print("Size of Original Train : ", original_train.shape)

original_train.rename(columns={'Gender': 'Sex'}, inplace=True)
train = pd.concat([train, original_train], ignore_index = True)
print("Size of Train : ", train.shape)


submit['Calories'] = 0

features = list(train.columns[:-1])
TARGET = train.columns[-1]
numeric_features = test.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = test.select_dtypes(include=['object']).columns.tolist()

# Check for missing values
print("Missing values in train set:")
print(train.isnull().sum())
print("Missing values in test set:")
print(test.isnull().sum())


def GBDT_preprocessing(features,train, test):
    # Fill missing values. -> No missing values in this dataset. 
    # Convert categorical variables to numerical.
    '''
    'Sex' : 'male' -> 1, 'female' -> 0
    '''
    for df in [train,test]:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # PCA to remove Covariance. -> Not useful in this case.
    # skewed data : Age, Weight : Log Transform to get normal. 
    for col in ['Weight']:
        train[col] = np.log1p(train[col])
        test[col] = np.log1p(test[col])
    
    # Negative skewed data : Body_Temp / Non-linear relationship with target : Duration, Heart_Rate, Body_Temp. -> binning.
    for col in ['Duration', 'Heart_Rate', 'Body_Temp']:
        train['BIN_' + col] = pd.cut(train[col], bins=10, labels=False).astype(int)
        test['BIN_' + col] = pd.cut(test[col], bins=10, labels=False).astype(int)
        features.append('BIN_' + col)
    
    # Add BMI Feature and BMI Categorical Feature.
    for df in [train, test]:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    features.append('BMI')

    for df in [train, test]:
        df['CAT_BMI'] = pd.cut(df['BMI'], bins=[-np.inf, 18.5, 23, 25, 29, np.inf], labels=[0,1,2,3,4]).astype(int)
    features.append('CAT_BMI')

    # Add Interaction Features.
    for comb in combinations(numeric_features, 2):
        if comb[0] != 'Calories' and comb[1] != 'Calories':
            train[f'{comb[0]}*{comb[1]}'] = train[comb[0]] * train[comb[1]]
            test[f'{comb[0]}*{comb[1]}'] = test[comb[0]] * test[comb[1]]
            features.append(f'{comb[0]}*{comb[1]}')

    # Domain-specific Interaction Features. 
    for df in [train, test]:
        df['Effort'] = (df['Duration'] * df['Heart_Rate']) / df['Weight']
        df['MET_proxy'] = df['Heart_Rate']*df['Body_Temp'] / df['Age']
    
    features.append('Effort')
    features.append('MET_proxy')
        
    # Add AgeSex Columns using LabelEncoder. 
    # Add Age, Sex, AgeSex Categorical columns.
    train['AgeSex'] = train['Age'].astype(str) + train['Sex'].astype(str)
    test['AgeSex'] = test['Age'].astype(str) + test['Sex'].astype(str)
    
    train['AgeSex'] = LabelEncoder().fit_transform(train['AgeSex'])
    test['AgeSex'] = LabelEncoder().fit_transform(test['AgeSex'])
    features.append('AgeSex')
    
    train['Age'] = LabelEncoder().fit_transform(train['Age'])
    test['Age'] = LabelEncoder().fit_transform(test['Age'])

    for col in [f for f in features if 'CAT_' in f or 'BIN_' in f or f in ['Sex', 'AgeSex']]:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # Change Target -> log1p
    '''
    TARGET : log1p transform to minimize RMSLE. (Regression model struggle to minimize RMSE)
    '''
    train[TARGET] = np.log1p(train[TARGET])
    return features, train, test


def LINEAR_preprocessing(features,train, test):
    # Fill missing values. -> No missing values in this dataset. 
    # Convert categorical variables to numerical.
    '''
    'Sex' : 'male' -> 1, 'female' -> 0
    '''
    for df in [train,test]:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Add 'Body_Temp','Duration' quadratic term. (Great Feature)
    train['quad_Body_Temp'] = train.Body_Temp ** 2
    test['quad_Body_Temp'] = test.Body_Temp ** 2

    train['quad_Duration'] = train.Duration ** 2
    test['quad_Duration'] = test.Duration ** 2
    features.append('quad_Body_Temp')
    features.append('quad_Duration')

    # PCA to remove Covariance. 
    # Heart_Rate, Duration : Log Transform. 
    train['log_Duration'] = np.log1p(train['Duration'].to_numpy())
    test['log_Duration'] = np.log1p(test['Duration'].to_numpy())
    features.append('log_Duration')
    
    train['log_Heart_Rate'] = np.log1p(train['Heart_Rate'].to_numpy())
    test['log_Heart_Rate'] = np.log1p(test['Heart_Rate'].to_numpy())
    features.append('log_Heart_Rate')
    
    # Fill Nan values with median.
    train = train.fillna(train.median())
    test = test.fillna(test.median())

    new_features_train = {}
    new_features_test = {}
    new_features = []
    # Add interaction terms. (level = 2, 3)
    
    for lvl in range(2,4):
        for comb in combinations(features, lvl):
            # Product
            new_col_name = ' * '.join(comb)
            new_features_train[new_col_name] = train[list(comb)].prod(axis=1)
            new_features_test[new_col_name] = test[list(comb)].prod(axis=1)
            new_features.append(new_col_name)
            
            # Division
            if lvl == 2 and 'Sex' not in comb:
                new_col_name = ' / '.join(comb)
                new_features_train[new_col_name] = train[comb[0]] / train[comb[1]]
                new_features_test[new_col_name] = test[comb[0]] / test[comb[1]]
                new_features.append(new_col_name)
    
    train = pd.concat([train, pd.DataFrame(new_features_train)], axis = 1)
    test = pd.concat([test, pd.DataFrame(new_features_test)], axis = 1)
    
    features += new_features

    # Standard Scaler
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    # Change Target -> log1p
    '''
    TARGET : log1p transform to minimize RMSLE. (Regression model struggle to minimize RMSE)
    '''
    train[TARGET] = np.log1p(train[TARGET])
    return features, train, test


xgb_param = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",
    "seed": 42,
    "max_depth": 9,
    "learning_rate": 0.01,
    "reg_alpha": 0.3293064317672393,
    "reg_lambda": 2.9844553203552104,
    "gamma": 0.013536383495642845,
    "colsample_bytree": 0.58120898538853,
    "colsample_bynode": 0.6335076477473434
}

lgbm_param = {
    "objective": "regression_l2",
    "metric": "rmse",
    "device_type": "cpu",
    "force_col_wise" : True,
    "seed": 42,
    "early_stopping_round": 100,
    "verbose": 1,
    "num_leaves" : 31,
    "learning_rate": 0.022179258204781553,
    "reg_alpha": 0.5,
    "reg_lambda": 0.7468104017512764,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9
}

cb_param = {
    'iterations': 15000,
    'learning_rate': 0.011217327467389394,
    'depth': 5,
    'l2_leaf_reg': 0.5,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 100,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 1000
}


predict = {'xgb' : submit.copy(), 'lgbm' : submit.copy(), 'cb' : submit.copy(), 'mlr' : submit.copy()}

def train_and_predict(model_name, features, train, test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        X_train, X_valid = train[features].iloc[train_idx], train[features].iloc[valid_idx]
        y_train, y_valid = train[TARGET].iloc[train_idx], train[TARGET].iloc[valid_idx]

        print(f"Fold {fold + 1}/{kf.get_n_splits()} for {model_name}")
        
        if model_name == 'xgb':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            model_instance = xgb.train(xgb_param, dtrain, num_boost_round=10000, evals=[(dvalid, 'valid')], verbose_eval=1000)
            y_pred = model_instance.predict(X_valid)
            predict['xgb'][TARGET] += np.expm1(model_instance.predict(xgb.DMatrix(test[features]))) / kf.get_n_splits()
            
            
        elif model_name == 'lgbm':
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=[col for col in features if 'BIN_' in col or 'CAT_' in col or col in ['Sex', 'AgeSex']])
            dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=[col for col in features if 'BIN_' in col or 'CAT_' in col or col in ['Sex', 'AgeSex']])
            model_instance = lgb.train(lgbm_param, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid])
            y_pred = model_instance.predict(X_valid)
            predict['lgbm'][TARGET] += np.expm1(model_instance.predict(test[features])) / kf.get_n_splits()


        elif model_name == 'cb':
            model_instance = cb.CatBoostRegressor(**cb_param, cat_features = [col for col in features if 'BIN_' in col or 'CAT_' in col or col in ['Sex', 'AgeSex']])
            model_instance.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=1000)
            y_pred = model_instance.predict(X_valid)
            predict['cb'][TARGET] += np.expm1(model_instance.predict(test[features])) / kf.get_n_splits()

        elif model_name == 'mlr':
            from sklearn.linear_model import LinearRegression
            model_instance = LinearRegression()
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_valid)
            predict['mlr'][TARGET] += np.expm1(model_instance.predict(test[features])) / kf.get_n_splits()
            
        rmsle_score = np.sqrt(np.mean((y_pred - y_valid)**2))
        print(f"Fold {fold +1} RMSLE : {rmsle_score}")
        rmsle.append(rmsle_score)
        
    print(f"Total RMSLE : {np.array(rmsle).mean()}")
    


print(f"Preprocessing for model...")
features_gbdt, train_gbdt, test_gbdt = GBDT_preprocessing(features.copy(), train.copy(), test.copy())
features_mlr, train_mlr, test_mlr = LINEAR_preprocessing(features.copy(), train.copy(), test.copy())

# I used another xgboost models for the final submission. If using this xgboost code, Score will be little bit higher. (0.0001?)
for model_name in ['cb', 'mlr','lgbm']:
    print(f"Training {model_name} model...")
    if model_name in ['xgb', 'lgbm', 'cb']:
        train_and_predict(model_name, features_gbdt, train_gbdt, test_gbdt)
    else:
        train_and_predict(model_name, features_mlr, train_mlr, test_mlr)


# I also used my single best xgboost model. (Referenced notebook : )
predict['xgb0'] = pd.read_csv('/kaggle/input/pl-s5e5-xgboost-lb0-0566/submission_xgb0.csv')

# weight calculated using cdeotte's GPU Hill Climbing Notebook
final_pred = predict['xgb0'][TARGET].values*0.47124 + predict['lgbm'][TARGET].values * 0.16 + predict['cb'][TARGET].values * 0.24276 + predict['mlr'][TARGET].values * 0.126

submit['Calories'] = final_pred
submit.to_csv(f'submission.csv', index=False)
print("Submission saved as submission.csv")

