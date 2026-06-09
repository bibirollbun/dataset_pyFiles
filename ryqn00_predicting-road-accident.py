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


train_pd = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train_pd.head()


numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']


import seaborn as sns
import matplotlib.pyplot as plt

for feature in train_pd.columns:
    if feature in numerical_features:
        plt.figure(figsize=(3, 2))
        sns.boxplot(data=train_pd, x=feature)  # kde + bins for smoother look
        plt.xlabel(feature)
        plt.ylabel("Counts")
        plt.title(f"Distribution of {feature}")
        plt.show()
        
    elif feature in categorical_features:
        plt.figure(figsize=(3, 2))
        sns.countplot(data=train_pd, x=feature)  # countplot instead of barplot
        plt.xlabel(feature)
        plt.ylabel("Counts")
        plt.title(f"Countplot of {feature}")
        plt.show()


for num_feature in numerical_features:
    plt.figure(figsize=(2, 2))
    sns.jointplot(data=train_pd, x='accident_risk', kind='hex',marginal_kws=dict(bins=30), y=num_feature)  # kde + bins for smoother look
    plt.xlabel(feature)
    plt.ylabel(num_feature)
    plt.title(f"Distribution of {feature}")
    plt.show()
    


feature_boxplot = categorical_features + ['num_reported_accidents', 'speed_limit', 'num_lanes']


for feature in feature_boxplot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_pd, x=feature, y='accident_risk') 
    plt.xlabel(feature)
    plt.ylabel('accident_risk')
    plt.title(f"Distribution of {feature}")
    plt.show()


train_pd.info()


from sklearn.model_selection import train_test_split

# Correct way: Separate features from target before splitting
X = train_pd.drop(columns=['accident_risk', 'id'], axis=1)  # Features only
y = train_pd['accident_risk']               # Target only

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)


X_train.shape, X_val.shape, y_train.shape, y_val.shape


X_train = pd.get_dummies(X_train, columns=categorical_features, prefix_sep='_', drop_first=False)
X_train.head()


X_val = pd.get_dummies(X_val, columns=categorical_features, prefix_sep='_', drop_first=False)
X_val.head()


import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import optuna
import warnings


# Let's use copy of training dataset
X_copy = X.copy()
X_copy = pd.get_dummies(X_copy, columns=categorical_features, prefix_sep='_', drop_first=False)
y_copy = y.copy()

# Suppress all warnings
warnings.filterwarnings('ignore')
'''
def objective_with_validation_split(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'num_leaves': trial.suggest_int('num_leaves', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'objective': 'regression',
        'metric': 'rmse',
        'subsample_freq': 1,
        'verbosity' : -1,
        'random_state': 42,
        'n_jobs': -1,  # Use all CPU cores instead of GPU
        # REMOVE these GPU parameters:
        # 'device': 'cuda',
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0,
    }

    model = lgb.LGBMRegressor(**params)

    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_copy, y_copy, test_size=0.2, random_state=42, shuffle=True
    )

    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    # Return negative RMSE (since Optuna maximizes, and we want to minimize RMSE)
    return -rmse

study_lgb = optuna.create_study(study_name='LGBMRegbase fine tuning', direction='maximize')
study_lgb.optimize(objective_with_validation_split, n_trials=1000)

# Get best parameters:
best_params_lgb = study_lgb.best_params
print(f"Best RMSE: {study_lgb.best_value:.4f}")
print("Best parameters:", best_params_lgb)
'''
'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store RMSE for each fold
fold_rmse = []

X = train_pd.drop(columns=["id", "accident_risk"])
y = train_pd["accident_risk"]

# 5-Fold CV
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    
    X_train = pd.get_dummies(X_train, columns=categorical_features, prefix_sep='_', drop_first=False)
    
    X_val = pd.get_dummies(X_val, columns=categorical_features, prefix_sep='_', drop_first=False)
    
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    XGBRegbase.fit(X_train, y_train)
    preds = XGBRegbase.predict(X_val)
        
    rmse = mean_squared_error(y_val, preds, squared=False)
    fold_rmse.append(rmse)
    
print(f"XGBRegbase mean RMSE: {np.mean(fold_rmse):.4f}")
'''


#import optuna.visualization as vis

#display(vis.plot_param_importances(study_lgb))
#display(vis.plot_optimization_history(study_lgb))


# XGB

'''XGBRegbase = xgb.XGBRegressor(n_estimators = 500,
                         max_depth = 3,
                         learning_rate = 0.1,
                         subsample = 0.8,
                         colsample_bytree = 0.8,
                         reg_lambda = 1,
                         reg_alpha = 0,
                         objective='reg:squarederror',
                         eval_metric='rmse',
                         random_state=42)

# Parameters used for v10
XGBRegoptuna1 = xgb.XGBRegressor(**{'n_estimators': 699, 
                                  'max_depth': 8, 
                                  'learning_rate': 0.010108705451891645, 
                                  'subsample': 0.6402784898024437, 
                                  'colsample_bytree': 0.9668274209321512, 
                                  'reg_lambda': 2.203450773706666,
                                  'reg_alpha': 0.002492531872446458}, 
                                   objective='reg:squarederror',
                                   eval_metric='rmse',
                                   random_state=42)
'''

# LGB
'''
LGBRegbase = lgb.LGBMRegressor(
    num_leaves=1000,
    max_depth=500,
    learning_rate=0.1,
    n_estimators=1000,
    min_data_in_leaf=100,
    colsample_bytree=0.8,  # Feature sampling
    subsample=0.8,         # Row sampling (bagging_fraction)
    subsample_freq=1,      # Enable bagging every iteration
    reg_lambda=1,
    reg_alpha=0,
    objective='regression',
    metric='rmse',         # Evaluation metric
    random_state=42        # Good practice for reproducibility
)
'''


# XGB
#XGBRegoptuna1.fit(X_copy, y_copy)

# LGB
'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store RMSE for each fold
fold_rmse = []

X = train_pd.drop(columns=["id", "accident_risk"])
y = train_pd["accident_risk"]

# 5-Fold CV
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    
    X_train = pd.get_dummies(X_train, columns=categorical_features, prefix_sep='_', drop_first=False)
    
    X_val = pd.get_dummies(X_val, columns=categorical_features, prefix_sep='_', drop_first=False)
    
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    LGBRegbase.fit(X_train, y_train)
    preds = LGBRegbase.predict(X_val)
        
    rmse = mean_squared_error(y_val, preds, squared=False)
    fold_rmse.append(rmse)
    
print(f"LGBRegbase mean RMSE: {np.mean(fold_rmse):.4f}")
'''


#X_val = pd.get_dummies(X_val, columns=categorical_features, prefix_sep='_', drop_first=False)
#y_pred_val = XGBRegbase.predict(X_val)

#print(f"XGBRegbase on test set mean RMSE: {mean_squared_error(y_val, y_pred_val, squared=False):.4f}")


best_params_lgb_v2 = {'n_estimators' : 5000 ,'num_leaves': 2320, 'max_depth': 7, 'learning_rate': 0.010261194451028046, 'subsample': 0.8334444883201919, 'min_data_in_leaf': 104, 'colsample_bytree': 0.9668215878324868, 'reg_lambda': 0.4362985902558426, 'reg_alpha': 1.4281065639641164}


#LGBRegbase.fit(X_copy, y_copy)
LGBReg_fine_tuned_v2 = lgb.LGBMRegressor(
    **best_params_lgb_v2,     
    objective='regression',
    metric='rmse',         # Evaluation metric      # Good practice for reproducibility
    n_jobs=-1,
    random_state = 42
)

LGBReg_fine_tuned_v2.fit(X_copy, y_copy)


test_pd = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
X_test = pd.get_dummies(test_pd, columns=categorical_features, prefix_sep='_', drop_first=False)
X_test = X_test.drop(columns=['id'])


X_test.head()


# Now predict
submission_lgb_v2 = LGBReg_fine_tuned_v2.predict(X_test)
# submission = XGBRegbase.predict(X_test)
submission = pd.DataFrame({'id': test_pd.id, 'accident_risk': submission_lgb_v2})
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()

