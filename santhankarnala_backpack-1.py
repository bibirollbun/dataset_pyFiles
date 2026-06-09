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


'''import pandas as pd
train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train.columns'''


#train2=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
#train2.columns


'''from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Brand']=le.fit_transform(train['Brand'])
train['Material']=le.fit_transform(train['Material'])
train['Size']=le.fit_transform(train['Size'])
train['Laptop Compartment']=le.fit_transform(train['Laptop Compartment'])
train['Waterproof']=le.fit_transform(train['Waterproof'])
train['Style']=le.fit_transform(train['Style'])
train['Color']=le.fit_transform(train['Color'])
train.info()
'''

#XGB doesn't accept label encoded values


'''test['Brand']=le.fit_transform(test['Brand'])
test['Material']=le.fit_transform(test['Material'])
test['Size']=le.fit_transform(test['Size'])
test['Laptop Compartment']=le.fit_transform(test['Laptop Compartment'])
test['Waterproof']=le.fit_transform(test['Waterproof'])
test['Style']=le.fit_transform(test['Style'])
test['Color']=le.fit_transform(test['Color'])
test.info()
'''


#making xgb to use the QuantileDMatrix
# Label Encoding and explicitly setting dtype to category
'''for col in ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']:
    train[col] = pd.Categorical(train[col])  # Convert to categorical
    test[col] = pd.Categorical(test[col]) # Convert to categorical'''


'''X=train.drop(columns=['id', 'Price'])
y=train['Price']'''


'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''


'''import numpy as np
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    tree_method="hist",
    device="cuda",
    enable_categorical=True  # This is the crucial addition!
)
xgb_model.fit(X,y)'''



'''predictions = xgb_model.predict(test.drop(columns=['id']))
submission = pd.DataFrame({'id': test['id'], 'Price': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")'''


'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler'''


pip install lightgbm==3.3.5 optuna==3.0.5


pip install pandas numpy scikit-learn lightgbm optuna


'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Convert categorical columns to 'category' dtype
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Separate features and target
X = train.drop(columns=['id', 'Price'])
y = train['Price']'''


X_train, X_val, y_train, y_val = train_test_'''split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset with free_raw_data=False
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, free_raw_data=False, reference=train_data)'''


'''def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,  # Suppress LightGBM logs
        'early_stopping_rounds': 50  # Add early stopping here
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=10)]  # Add verbosity here
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    return np.sqrt(mean_squared_error(y_val, val_preds))'''



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Print column names to verify
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Convert categorical columns to 'category' dtype
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_cols:
    if col in train.columns:
        train[col] = train[col].astype('category')
    if col in test.columns:
        test[col] = test[col].astype('category')

# Separate features and target
X = train.drop(columns=['id', 'Price'])
y = train['Price']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, free_raw_data=False, reference=train_data)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,  # Suppress LightGBM logs
        'early_stopping_rounds': 50  # Add early stopping here
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=10)]  # Add verbosity here
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    return np.sqrt(mean_squared_error(y_val, val_preds))

# Run Optuna optimization
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# Best parameters
best_params = study.best_params
print(f'Best parameters: {best_params}')

# Train final model with best parameters
final_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.log_evaluation(period=10)]  # Add verbosity here
)

# Evaluate on validation set
val_predictions = final_model.predict(X_val, num_iteration=final_model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Validation RMSE: {val_rmse}')

# Make predictions on test data
test_predictions = final_model.predict(test.drop(columns=['id']), num_iteration=final_model.best_iteration)

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


pip install pandas numpy scikit-learn lightgbm optuna


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Print column names to verify
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Rename categorical columns to remove spaces
train = train.rename(columns=lambda x: x.replace(" ", "_"))
test = test.rename(columns=lambda x: x.replace(" ", "_"))

# Define categorical columns after renaming
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop_Compartment', 'Waterproof', 'Style', 'Color']

# Convert categorical columns to 'category' dtype
for col in categorical_cols:
    if col in train.columns:
        train[col] = train[col].astype('category')
    if col in test.columns:
        test[col] = test[col].astype('category')

# Separate features and target
X = train.drop(columns=['id', 'Price'])
y = train['Price']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, free_raw_data=False, reference=train_data)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,  # Suppress LightGBM logs
        'early_stopping_rounds': 50  # Add early stopping here
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=10)]  # Add verbosity here
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    return np.sqrt(mean_squared_error(y_val, val_preds))

# Run Optuna optimization
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# Best parameters
best_params = study.best_params
print(f'Best parameters: {best_params}')

# Train final model with best parameters
final_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.log_evaluation(period=10)]  # Add verbosity here
)

# Evaluate on validation set
val_predictions = final_model.predict(X_val, num_iteration=final_model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Validation RMSE: {val_rmse}')

# Make predictions on test data
test_predictions = final_model.predict(test.drop(columns=['id']), num_iteration=final_model.best_iteration)

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")





