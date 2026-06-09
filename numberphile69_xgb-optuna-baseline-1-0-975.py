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


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
original=pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")


train=train.drop("id",axis=1)


original


train


train=pd.concat([train,original],axis=0)


train=train.dropna()


train


train


import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load your data - replace these with your actual data paths
train_df = train # Contains target variable
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')    # Doesn't contain target variable

# Separate features and target
X = train_df.drop('Personality', axis=1)
y = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})  # Convert to binary

# List of categorical features
categorical_features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
]

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)
test_preprocessed = preprocessor.transform(test_df)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': 42,
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

# Optimize using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

# Get best parameters
best_params = study.best_params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
})

# Train final model on full training data with best parameters
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_preprocessed, y)

# Make predictions on test set
test_predictions = final_model.predict(test_preprocessed)
test_probabilities = final_model.predict_proba(test_preprocessed)[:, 1]  # Probability of being Extrovert

# Convert predictions back to original labels
predicted_labels = np.where(test_predictions == 1, 'Extrovert', 'Introvert')

# Create output DataFrame
results_df = test_df.copy()
results_df['Predicted_Personality'] = predicted_labels
results_df['Extrovert_Probability'] = test_probabilities



results_df


test_df


sub=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


sub=sub.drop("Personality",axis=1)


sub["Personality"]=results_df["Predicted_Personality"]


sub


sub.to_csv("submission23.csv",index=False)






















