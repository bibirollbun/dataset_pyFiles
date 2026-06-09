# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from xgboost import XGBClassifier

import matplotlib.pyplot as plt 
import seaborn as sns

set_config(display='diagram')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_file = '/kaggle/input/playground-series-s5e8/train.csv'
X = pd.read_csv(train_file)
y = X['y']

X.drop(columns=['y'], inplace=True)

# drop the id column if it exists
if 'id' in X.columns:
    X.drop(columns=['id'], inplace=True)

X_orig = X.copy()
# print missing or null values for every column
if X.isnull().values.any():
    print("There are missing values in the data.")
    print("Missing values in training data by column:\n", X.isnull().sum())
    exit(1)


print("Shape of the data:", X.shape)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist() 
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# find columns with zero variance
zero_variance_cols = X.columns[X.nunique() <= 1]
print("Columns with zero variance:", zero_variance_cols.tolist())
# drop columns with zero variance  
X.drop(columns=zero_variance_cols, inplace=True)

# find columns with high correlation only for numerical columns
# get numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numerical_cols]

corr_matrix = X.corr().abs()
high_corr_var = np.where(corr_matrix > 0.95)
high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
print("Columns with high correlation:", high_corr_var)

# draw a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix Heatmap") 
plt.show()

# restore X to original
X = X_orig.copy()
X.drop(columns=zero_variance_cols, inplace=True)


# Define separate pipelines for numerical and categorical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include='object').columns

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # constant strategy is the alternate
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


models = [
    {
        'name': 'gradient_boosting',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
            ('gb', GradientBoostingClassifier())
        ]),
        'param_grid': {
            'gb__n_estimators': [50, 100, 200],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 6, 10],
            'gb__subsample': [0.8, 1.0]
        },
        "results": {
            "best_score": 0.0,  # Placeholder for best score
            "best_params" : None # Placeholder for best score
        },
        "enabled": False
    },
    {
        'name': 'svm',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
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
        "enabled": False
    },
    {
        'name': 'random_forest',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
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
        "enabled": False    
    },
    {
        'name': 'decision_tree',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
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
        "enabled": False
    },
    {
        'name': 'xgboost',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
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
        "enabled": False
    },
    {
        'name': 'logistic_regression',
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
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
            ("preprocessor", preprocessor),
            ('gnb', GaussianNB())
        ]),
        'param_grid': {},  # No hyperparameters to tune for GaussianNB
        "results": {
            "best_score": 0.0,
            "best_params": None
        },
        "enabled": False
    }
]


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")


# Initialize a variable to keep track of the best score as negative infinity
best_score = float('-inf')

print("Starting model training and hyperparameter tuning...\n")
print(f"Total models to evaluate: {len(models)}")
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
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
    if not model['enabled']:
        continue
    print(f"\nResults for {model['name']}:")
    print(f"Best Score: {model['results']['best_score']}")
    print(f"Best Parameters: {model['results']['best_params']}")    
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
# print(f"\nBest model: {best_model} with parameters: {best_params} and score: {best_score}") 


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
test_file = '/kaggle/input/playground-series-s5e8/test.csv'
test_data_raw = pd.read_csv(test_file)

# save the ids
ids = test_data_raw['id']

# Remove 'id' column if present before transforming
test_data_raw = test_data_raw.drop(columns=['id'])

# Predict probabilities
y_test_pred = final_model.predict_proba(test_data_raw)[:, 1]  # Use [:, 1] for positive class

# Save to CSV
output_file = 'submission.csv'
submission_df = pd.DataFrame({'id': ids, 'y': y_test_pred}) if ids is not None else pd.DataFrame({'y': y_test_pred})
submission_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")


# extract the best features from logistic regression model
if best_model['name'] == 'logistic_regression':
    lr_model = best_model_pipeline.named_steps['lr']
    preprocessor = best_model_pipeline.named_steps['preprocessor']
    
    # Get feature names after preprocessing
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numerical_cols, ohe_feature_names])
    
    # Get coefficients
    coefficients = lr_model.coef_[0]
    
    # Create a DataFrame for better visualization
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    # print("\nTop features based on absolute coefficient values:")
    # print(coef_df.head(10))
    
    # Plot the coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=coef_df.head(10), palette='viridis')
    plt.title('Top 10 Features by Absolute Coefficient Value')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature') 
    plt.show()
elif best_model['name'] in ['random_forest', 'decision_tree', 'gradient_boosting', 'xgboost']:
    feature_names = final_model.named_steps['preprocessor'].get_feature_names_out()
    if best_model['name'] == 'random_forest':
        importances = final_model.named_steps['rf'].feature_importances_
    elif best_model['name'] == 'decision_tree':
        importances = final_model.named_steps['dt'].feature_importances_
    elif best_model['name'] == 'gradient_boosting':
        importances = final_model.named_steps['gb'].feature_importances_
    elif best_model['name'] == 'xgboost':
        importances = final_model.named_steps['xgb'].feature_importances_
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title(f'Top 20 Feature Importances in {best_model["name"].replace("_", " ").title()}')
    plt.show()
    

