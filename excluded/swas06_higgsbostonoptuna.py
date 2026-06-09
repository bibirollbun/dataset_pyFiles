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


df_train = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/test.csv")


df_train.head(3)


X=df_train.drop("label",axis=1)
y=df_train['label']


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

# Fit and transform the data (scale the features)
X_scaled = scaler.fit_transform(X)

# If you want to check the scaled data:
print(X_scaled)


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameters to tune
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
    }
    
    # Create the XGBoost model
    model = xgb.XGBClassifier(**param)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # Calculate the AUC score
    auc = roc_auc_score(y_test, y_pred)
    return auc

# Create the Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the corresponding AUC score
print('Best hyperparameters:', study.best_params)
print('Best AUC score:', study.best_value)


best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42)

final_model.fit(X_scaled, y)


df_test_scaled = scaler.fit_transform(df_test)


test_pred_proba = final_model.predict_proba(df_test_scaled)[:, 1]


submission_df = pd.read_csv('/kaggle/input/higgs-boson-detection-2025/sample_submission.csv')


submission_df["Predicted"] = test_pred_proba
submission_df['Id'] = submission_df['Id'].astype(np.int64)
submission_df['Id'] = submission_df['Id'].apply(lambda x: f"{float(x):.18e}")
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

