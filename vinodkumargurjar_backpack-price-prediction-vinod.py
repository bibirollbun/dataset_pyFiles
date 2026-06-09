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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_extra_train=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


df_train.head(2)


df_extra_train.head(2)


df_test.head(2)


sample_submission.head(5)


df_train.info()


df_test.info()


df_train.isnull().sum()


df_test.isnull().sum()


len(df_train["id"]),len(df_test["id"])



((df_train.isnull().sum())/(len(df_train["id"])))*100



((df_test.isnull().sum())/(len(df_test["id"])))*100



df_train.drop(columns="id", axis=1, inplace=True)
df_test.drop(columns="id", axis=1, inplace=True)


df_train.head(3)



train_cat_columns=[]
for i in df_train.columns:
    if df_train[i].dtypes=='O':
        train_cat_columns.append(i)
train_cat_columns
    


test_cat_columns=[]
for i in df_test.columns:
    if df_test[i].dtypes=='O':
        test_cat_columns.append(i)
test_cat_columns


df_train[train_cat_columns].nunique()


df_test[test_cat_columns].nunique()


for column in df_train.columns:
    if df_train[column].dtype == 'object':
        # Fill with mode for object columns
        mode_value = df_train[column].mode()[0]  # Get the mode and take the first one if there are multiple
        df_train[column].fillna(mode_value, inplace=True)
    elif df_train[column].dtype in ['int64', 'float64']:
        # Fill with mean for numeric columns
        mean_value = df_train[column].mean()
        df_train[column].fillna(mean_value, inplace=True)


df_train.isnull().sum()


for column in df_test.columns:
    if df_test[column].dtype == 'object':
        # Fill with mode for object columns
        mode_value = df_test[column].mode()[0]  # Get the mode and take the first one if there are multiple
        df_test[column].fillna(mode_value, inplace=True)
    elif df_test[column].dtype in ['int64', 'float64']:
        # Fill with mean for numeric columns
        mean_value = df_test[column].mean()
        df_test[column].fillna(mean_value, inplace=True)


df_test.isnull().sum()



df_train1=df_train.copy()
df_test1=df_test.copy()


from sklearn.preprocessing import LabelEncoder

# Loop through all columns in df_train1
for column in df_train1.columns:
    if df_train1[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()  # Create a LabelEncoder object
        df_train1[column] = le.fit_transform(df_train1[column].astype(str))  # Fit and transform the column


# Loop through all columns in df_train
for column in df_test1.columns:
    if df_test1[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()  # Create a LabelEncoder object
        df_test1[column] = le.fit_transform(df_test1[column].astype(str))  # Fit and transform the column


X=df_train.drop("Price",axis=1)
y=df_train["Price"]


X1=df_train1.drop("Price",axis=1)
y1=df_train1["Price"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=104,shuffle=True)


X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size=0.15,random_state=104,shuffle=True)


X_train.shape,X_test.shape,y_train.shape,y_test.shape


X_train.head(2)


X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape


X_train1.head(2)


# import catboost
# import optuna
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from catboost import CatBoostRegressor
# # Define the objective function for Optuna
# def objective(trial):
#     # Suggest hyperparameters
#     params = {
#         "iterations": trial.suggest_int("iterations", 500, 3000),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
#         "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
#         "random_strength": trial.suggest_float("random_strength", 0.1, 10),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#     }

#     # Train CatBoostRegressor
#     model = CatBoostRegressor(**params, loss_function="RMSE", verbose=0, random_state=42,
#                               cat_features=train_cat_columns)
#     model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=0)

#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)

#     return rmse

# # Run optimization
# study = optuna.create_study(direction="minimize")  # Minimize RMSE
# study.optimize(objective, n_trials=50, timeout=600)  # 50 trials or 10 minutes max

# # Best hyperparameters
# print("Best hyperparameters:", study.best_params)


params={'iterations': 2092, 'depth': 5, 'learning_rate': 0.0725226271750525,
        'l2_leaf_reg': 1.8024960853147594, 'subsample': 0.5596227947364816, 
        'colsample_bylevel': 0.7907904989487283, 'boosting_type': 'Ordered',
        'random_strength': 2.3486728861546067, 
        'bagging_temperature': 0.03561264508908479, 
        'border_count': 198}


import catboost
from catboost import CatBoostRegressor
cbr=CatBoostRegressor(**params, loss_function="RMSE", verbose=0, random_state=42,
                              cat_features=train_cat_columns)
cbr.fit(X_train,y_train)


# import optuna
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# def objective(trial):
#     # Suggest hyperparameters
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 500),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 10),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
#     }

#     # Initialize XGBoost model with suggested hyperparameters
#     model = xgb.XGBRegressor(objective="reg:squarederror", **params)
    
#     # Train the model
#     model.fit(X_train1, y_train1)
    
#     # Make predictions
#     y_pred = model.predict(X_test1)
    
#     # Calculate MSE
#     mse = mean_squared_error(y_test1, y_pred)
#     rmse=np.sqrt(mse)
    
#     return rmse  # Optuna minimizes this
# study = optuna.create_study(direction="minimize")  # We minimize MSE
# study.optimize(objective, n_trials=50)  # Run 50 trials

# # Best hyperparameters found
# best_params = study.best_params
# print("Best Hyperparameters:", best_params)



best_params={'n_estimators': 462, 'max_depth': 3, 
             'learning_rate': 0.022785973289324142, 'subsample': 0.7090057757902789, 
             'colsample_bytree': 0.9619326812453037, 'gamma': 8.666042435197538,
             'reg_alpha': 5.468333438373428, 
             'reg_lambda': 7.674526662274267}


import xgboost as xgb
xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", **best_params)

# Train the model
xgb_regressor.fit(X_train1, y_train1)


from sklearn.metrics import mean_squared_error
prediction_cbr=cbr.predict(X_test)
mse_cbr=mean_squared_error(y_test,prediction_cbr)
Root_mean_squared_error_cbr=np.sqrt(mse_cbr)
print("Root mean_squared_error Catboost Regressor is ",Root_mean_squared_error_cbr)


prediction_xgb=xgb_regressor.predict(X_test1)
mse_xgb=mean_squared_error(y_test1,prediction_xgb)
Root_mean_squared_error_xgb=np.sqrt(mse_xgb)
print("Root mean_squared_error XGBOOST Regressor is ",Root_mean_squared_error_xgb)


final_result=xgb_regressor.predict(df_test1)



sample_submission["Price"]=final_result



sample_submission.to_csv('submission.csv',index=False)



sample_submission.head(5)







