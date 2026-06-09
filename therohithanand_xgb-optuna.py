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


# import necessary libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import optuna
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer  
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import IterativeImputer


# Read data and remove unnecesary features

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv') 

ids = test['id']
train.drop(columns='id',inplace=True)
test.drop(columns='id',inplace=True)


# Take a look at the statistics of the dataset

print("Dataset shape: ",train.shape)

print("Features: \n",train.columns)


print("\nNumerical Features Summary:")
display(train.describe())

print("\nFirst 10 Rows of the Dataset:")
display(train.head(10))


# Seperate numerical and categorical data

numeric = test.select_dtypes(include=['number']).columns
categorical = test.select_dtypes(exclude=['number']).columns


# Check if there are any null values in the dataset 

train.isnull().sum()


# Use iterative imputer to impute missing data

def preprocessing(data, numeric, categorical):
    # --- Iterative Imputation for Numeric Columns ---
    imputer = IterativeImputer(random_state=42)
    
    # Only apply to numeric subset
    numeric_data = data[numeric]
    imputed_numeric = imputer.fit_transform(numeric_data)
    data[numeric] = pd.DataFrame(imputed_numeric, columns=numeric, index=data.index)

    # --- Mode Imputation for Categorical Columns ---
    for col in categorical:
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            print(f"Warning: No mode found for column {col}, possibly all values are NaN.")

    return data

# Example usage:
numeric = train.select_dtypes(include=np.number).columns.tolist()
categorical = test.select_dtypes(exclude=np.number).columns.tolist()

train = preprocessing(train, numeric, categorical)
test = preprocessing(test, numeric, categorical)


# Visualize relation between numeric data

sns.pairplot(train[numeric],corner=True)


# Visualize heatmap 

sns.heatmap(train[numeric].corr(),annot=True)


# One Hot Encoding

def one_hot_encode(train, test, categorical):
    # Create OneHotEncoder instance
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Fit encoder on training categorical columns
    encoder.fit(train[categorical])

    # Transform both train and test
    train_encoded = encoder.transform(train[categorical])
    test_encoded = encoder.transform(test[categorical])

    # Convert to DataFrames with appropriate column names
    encoded_columns = encoder.get_feature_names_out(categorical)
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns, index=train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns, index=test.index)

    # Drop original categorical columns and concatenate encoded ones
    train = train.drop(columns=categorical).join(train_encoded_df)
    test = test.drop(columns=categorical).join(test_encoded_df)

    return train, test, encoder


train, test, one_hot_encoder = one_hot_encode(train, test, categorical)


# Label Encoding for target variable 

target_le = LabelEncoder()
train['Personality'] = target_le.fit_transform(train['Personality'])


# Prepare dataset for training

X = train.drop(columns=['Personality'])
y = train['Personality']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.25)


import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score
import numpy as np

xgb.set_config(verbosity=0)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    param = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3),
        'max_depth': trial.suggest_int("max_depth", 3, 30),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 30),
        'gamma': trial.suggest_float("gamma", 0, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        'n_estimators': trial.suggest_int("n_estimators", 10, 1000)
    }

    model = xgb.XGBClassifier(**param)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300, timeout=4000)



best_params = study.best_params

# Add fixed parameters separately
fixed_params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train)),
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0
}

# Merge both dictionaries
full_params = {**best_params, **fixed_params}

# Create and train the model
model = xgb.XGBClassifier(**full_params)


model.fit(X_train,y_train)


y_pred = model.predict(X_val)


acc = accuracy_score(y_pred,y_val)
acc


y_pred_test = model.predict(test)
y_pred_test = target_le.inverse_transform(y_pred_test)

sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
sub['id'] = ids
sub['Personality'] = y_pred_test
sub.to_csv('submission.csv',index=False)




