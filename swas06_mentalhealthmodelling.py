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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb


df_train = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")


df_train.columns = df_train.columns.str.replace(" ","_").str.lower()
df_test.columns =df_test.columns.str.replace(" ","_").str.lower()


df_train.columns


categorical_features =['name', 'gender', 'city', 'working_professional_or_student',
       'profession', 'sleep_duration', 'dietary_habits', 'degree',
       'have_you_ever_had_suicidal_thoughts_?',
       'family_history_of_mental_illness']


numeric_features = ['id', 'age', 'academic_pressure', 'work_pressure', 'cgpa',
       'study_satisfaction', 'job_satisfaction', 'work/study_hours',
       'financial_stress' ]


X = df_train.drop('depression',axis=1)
y = df_train['depression']



# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Classifiers to try
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over classifiers and evaluate using accuracy
for name, classifier in classifiers.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Predict the target
    y_pred = pipeline.predict(X_test)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def objective(trial):
    # Hyperparameters to tune
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", random_state=42)
    
    # Create the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Make sure 'preprocessor' is defined
        ('classifier', model)
    ])

    
    # Fit the pipeline (this will fit both preprocessor and classifier)
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    preds = pipeline.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, preds)
    
    # Return accuracy (Optuna will maximize it)
    return accuracy

# Create study object
study = optuna.create_study(direction="maximize")

# Optimize the objective function (50 trials)
study.optimize(objective, n_trials=50)

# Output the best hyperparameters and the best score
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)

# Train the best model with optimal hyperparameters
best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)


pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Make sure 'preprocessor' is defined
        ('classifier', best_model)
    ])
pipeline.fit(X_train, y_train)
# Test on the test set
best_preds = pipeline.predict(X_test)
final_accuracy = accuracy_score(y_test, best_preds)

print("Final Test Accuracy:", final_accuracy)


y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s4e11/sample_submission.csv')

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

