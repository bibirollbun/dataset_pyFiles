# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

_preprocess = {}
PP_VERSION = 1
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def _preprocess_v1(_df) -> pd.DataFrame: 
    df = _df.copy()
    # Fix missing values in numerical columns with median
    # use simple imputer for numerical columns
    simple_imputer = SimpleImputer(strategy='median')
    simple_imputer.fit(df.select_dtypes(include=['int64', 'float64']))
    df[df.select_dtypes(include=['int64', 'float64']).columns] = simple_imputer.transform(df.select_dtypes(include=['int64', 'float64']))
    
    # Fix missing values in categorical columns with One Hot Encoding
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include='object').columns

    # Apply OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])

    # Convert encoded features to DataFrame
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    # Drop original categorical columns and concatenate encoded columns
    df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df_encoded

def _preprocess_v2(_df) -> pd.DataFrame:
    df = _df.copy()
    
    # Fix missing values in numerical columns with simple imputer
    imputer = SimpleImputer(strategy='median')
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = imputer.fit_transform(df[[col]])

    # map the column Drained_after_socializing and Stage_fear between 0 and 1
    categorical_cols = ['Drained_after_socializing', 'Stage_fear']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            # fill missing values in these columns with the most frequent value
            df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

_preprocess[1] = _preprocess_v1
_preprocess[2] = _preprocess_v2



train_file = '/kaggle/input/playground-series-s5e7/train.csv'
X = pd.read_csv(train_file)

y = X['Personality'].map({'Introvert': 0, 'Extrovert': 1})

X.drop(columns=['Personality'], inplace=True)

# Fill missing values in training data
X = _preprocess[PP_VERSION](X)

# print missing or null values for every column
if X.isnull().values.any():
    print("There are missing values in the data.")
    print("Missing values in training data by column:\n", X.isnull().sum())
    exit(1)


# Combine pipelines and param_grids into a single structure
models = [
    {
        'name': 'svc',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ]),
        'param_grid': {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['linear', 'rbf'],
            'svc__gamma': ['scale', 'auto']
        },
        "results": {
            "best_score": 0.0,  # Placeholder for best score
            "best_params" : None # Placeholder for best score
        },
        "enabled": True
    },
    {
        'name': 'random_forest',
        'pipeline': Pipeline([
            ('rf', RandomForestClassifier())
        ]),
        'param_grid': {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 10, 20],
            'rf__random_state': [42]
        },
        "results": {
            "best_score": 0.0,  # Placeholder for best score
            "best_params" : None # Placeholder for best score
        },
        "enabled": True    },
    {
        'name': 'decision_tree',
        'pipeline': Pipeline([
            ('dt', DecisionTreeClassifier())
        ]),
        'param_grid': {
            'dt__criterion': ['gini', 'entropy'],
            'dt__max_depth': [None, 10, 20],
            'dt__random_state': [42]
        },
        "results": {
            "best_score": 0.0,  # Placeholder for best score
            "best_params" : None # Placeholder for best score
        },
        "enabled": True
    },
    {
    'name': 'xgboost',
    'pipeline': Pipeline([
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]),
    'param_grid': {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 6, 10],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    },
    "results": {
        "best_score": 0.0,
        "best_params": None
    },
    "enabled": True
    },
    {
        'name': 'logistic_regression',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'param_grid': {
            'lr__C': [0.01, 0.1, 1, 10],
            'lr__solver': ['lbfgs', 'liblinear']
        },
        "results": {
            "best_score": 0.0,
            "best_params": None
        },
        "enabled": True
    },
    {
        'name': 'gaussian_nb',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('gnb', GaussianNB())
        ]),
        'param_grid': {},  # No hyperparameters to tune for GaussianNB
        "results": {
            "best_score": 0.0,
            "best_params": None
        },
        "enabled": True
    }
]


# Initialize a variable to keep track of the best score as negative infinity
best_score = float('-inf')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Run grid search for each model
for model in models:
    if not model['enabled']:
        print(f"Skipping {model['name']} as it is disabled.")
        continue
    print(f"\nRunning GridSearchCV for {model['name']}...")
    grid_search = GridSearchCV(model['pipeline'], model['param_grid'], cv=5, scoring='accuracy', n_jobs=-1, refit=True)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model['name']}:", grid_search.best_params_)
    print(f"Best cross-validation score for {model['name']}:", grid_search.best_score_)
    
    # Update the model's pipeline with the best estimator
    model['pipeline'] = grid_search.best_estimator_
    # Update the model's results with the best score and parameters
    model['results']['best_score'] = grid_search.best_score_   
    model['results']['best_params'] = grid_search.best_params_   



# Print the results for each model
for model in models:
    # print(f"\nResults for {model['name']}:")
    # print(f"Best Score: {model['results']['best_score']}")
    # print(f"Best Parameters: {model['results']['best_params']}")
    
    # Update the global best score if this model's score is higher
    if model['results']['best_score'] > best_score:
        best_score = model['results']['best_score']
        best_model = model['name']
        best_params = model['results']['best_params']

model_names = [model['name'] for model in models]
scores = [model['results']['best_score'] for model in models]

plt.figure(figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.xticks(rotation=45)
plt.plot(model_names, scores, marker='o', color='skyblue', label='Model Score')
# Annotate each point with its scaled score value
for i, score in enumerate(scores):
    plt.text(model_names[i], score, f"{score:.6f}", ha='center', va='bottom', fontsize=10, color='blue')

# plt.axhline(y=best_score, color='r', linestyle='--', label='Best Score')
plt.legend()
# Print the best model and its parameters
print(f"\nBest model: {best_model} with parameters: {best_params} and score: {best_score}") 


# print the best model and its parameters
best_model = max(models, key=lambda x: x['results']['best_score']) 
print(f"\nBest model: {best_model['name']}")
best_model_pipeline = best_model['pipeline']
print(f"Best parameters: {best_model['results']['best_params']}")

final_model = best_model_pipeline.fit(X, y)
# Evaluate the final model on the test set
y_pred = final_model.predict(X)

print(f"Final model accuracy on train set: {accuracy_score(y, y_pred)}") 


# Predict on the test set
test_file = '/kaggle/input/playground-series-s5e7/test.csv'  
test_data = pd.read_csv(test_file)
test_data = _preprocess[PP_VERSION](test_data)

y_test_pred = final_model.predict(test_data)

y_test_pred = pd.Series(y_test_pred).map({0: 'Introvert', 1: 'Extrovert'})

# write the predictions to a CSV file
output_file = 'submission.csv'
submission_df = pd.DataFrame({'id': test_data['id'], 'Personality': y_test_pred})
submission_df.to_csv(output_file, index=False)
# Print the file path of the output file
print(f"Predictions saved to {output_file}")

