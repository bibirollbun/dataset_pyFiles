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


import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test =pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']
df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']


df_train['temp_from_dewpoint'] = df_train['temparature'] - df_train['dewpoint']
df_test['temp_from_dewpoint'] = df_test['temparature'] - df_test['dewpoint']


df_train.head(3)


df_train = df_train.drop(['mintemp','maxtemp','dewpoint'],axis="columns")
df_test = df_test.drop(['mintemp','maxtemp','dewpoint'],axis="columns")


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X=df_train.drop('rainfall',axis=1)
y=df_train['rainfall']


# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data (scale the features)
X_scaled = scaler.fit_transform(X)

# If you want to check the scaled data:
print(X_scaled)


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Compute class weights for imbalance handling (for models like Logistic Regression, Random Forest, SVM)
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Define the objective function
def objective(trial):
    # Choose the classifier (randomly or manually)
    model_type = trial.suggest_categorical('model_type', ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost'])

    # Hyperparameters for Logistic Regression
    if model_type == 'LogisticRegression':
        model = LogisticRegression(class_weight='balanced', max_iter=trial.suggest_int('max_iter', 100, 1000))
        param = {
            'C': trial.suggest_loguniform('C', 1e-5, 1e2),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }
        model.set_params(**param)

    # Hyperparameters for Random Forest
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model.set_params(**param)

    # Hyperparameters for Support Vector Machine
    elif model_type == 'SVM':
        model = SVC(class_weight='balanced', probability=True, random_state=42)
        param = {
            'C': trial.suggest_loguniform('C', 1e-5, 1e2),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e1)
        }
        model.set_params(**param)

    # Hyperparameters for XGBoost
    elif model_type == 'XGBoost':
        scale_pos_weight = len(y_train) / (2.0 * sum(y_train))  # for binary classification
        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
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
        model.set_params(**param)

    # Perform cross-validation to evaluate the model
    auc_score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return auc_score

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and corresponding AUC score
print('Best Hyperparameters:', study.best_params)
print('Best AUC score:', study.best_value)


best_params = {
    'model_type': 'LogisticRegression',  # You can ignore this in fit, it's just for Optuna's objective function
    'max_iter': 566,
    'C': 0.05521880917466241,
    'solver': 'liblinear'
}
model = LogisticRegression(
    class_weight='balanced',
    max_iter=best_params['max_iter'],
    C=best_params['C'],
    solver=best_params['solver'],
    random_state=42  # You can set the random_state for reproducibility
)



model.fit(X_scaled, y)


df_test_scaled = scaler.fit_transform(df_test)


test_pred_proba = model.predict_proba(df_test_scaled)[:, 1]


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission_df = pd.DataFrame({'id': df_test.id, 'rainfall': test_pred_proba})

submission_df.to_csv('submission.csv', index=False)
submission_df.head(10)

