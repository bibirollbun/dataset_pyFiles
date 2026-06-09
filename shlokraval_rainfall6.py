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


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import joblib


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# Feature and target selection
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])


# Compute feature correlation with target
correlation = X.corrwith(y).abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.05].index.tolist()
mandatory_features = ['maxtemp', 'mintemp', 'humidity', 'dewpoint']  # Add critical features
selected_features = list(set(selected_features + mandatory_features))  
X = X[selected_features]
X_test = X_test[selected_features]


# Identify numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['winddirection']


print(train.columns)  # Check exact column names



# Feature Engineering
train['temp_diff'] = train['maxtemp'] - train['mintemp']
train['humidity_dewpoint'] = train['humidity'] * train['dewpoint']
test['temp_diff'] = test['maxtemp'] - test['mintemp']
test['humidity_dewpoint'] = test['humidity'] * test['dewpoint']

# Feature Selection
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])

correlation = X.corrwith(y).abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.05].index.tolist()
X = X[selected_features]
X_test = X_test[selected_features]



# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])


print("Train Columns:", X.columns.tolist())
print("Test Columns:", X_test.columns.tolist())



# Model Training with Hyperparameter Tuning
models = {
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.05, random_state=42, verbose=0)
}

best_model = None
best_score = 0

for name, model in models.items():
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', BayesSearchCV(model, {
            'classifier__learning_rate': (0.01, 0.2, 'log-uniform'),
            'classifier__n_estimators': (50, 300)
        }, n_iter=10, cv=3, scoring='roc_auc', random_state=42))
    ])
    
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_pred)
    print(f'{name} AUC-ROC Score: {auc_score:.4f}')
    
    if auc_score > best_score:
        best_score = auc_score
        best_model = clf


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import optuna
import joblib

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Feature and target selection
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])

# Compute feature correlation with target
correlation = X.corrwith(y).abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.05].index.tolist()
X = X[selected_features]
X_test = X_test[selected_features]

# Identify numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['winddirection'] if 'winddirection' in X.columns else []

# Feature Engineering
if {'maxtemp', 'mintemp'}.issubset(X.columns):
    X['temp_diff'] = X['maxtemp'] - X['mintemp']
    X_test['temp_diff'] = X_test['maxtemp'] - X_test['mintemp']
    num_features.append('temp_diff')

if {'humidity', 'dewpoint'}.issubset(X.columns):
    X['humidity_dewpoint'] = X['humidity'] * X['dewpoint']
    X_test['humidity_dewpoint'] = X_test['humidity'] * X_test['dewpoint']
    num_features.append('humidity_dewpoint')

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Hyperparameter Optimization with Optuna
def objective(trial):
    model_type = trial.suggest_categorical("model", ["XGBoost", "LightGBM", "CatBoost"])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "random_state": 42
    }
    
    if model_type == "XGBoost":
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "LightGBM":
        model = LGBMClassifier(**params)
    else:
        model = CatBoostClassifier(iterations=params.pop("n_estimators"), **params, verbose=0)
    
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
best_model_type = best_params.pop("model")

if best_model_type == "XGBoost":
    best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
elif best_model_type == "LightGBM":
    best_model = LGBMClassifier(**best_params, random_state=42)
else:
    best_model = CatBoostClassifier(iterations=best_params.pop("n_estimators"), **best_params, random_state=42, verbose=0)

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

clf.fit(X, y)

# Save best model
joblib.dump(clf, 'best_rainfall_model.pkl')

# Generate Predictions for Test Set
test_preds = clf.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'id': test['id'], 'rainfall probability': test_preds})
submission.to_csv('submission.csv', index=False)

print('Best Model Saved and Predictions Generated!')

