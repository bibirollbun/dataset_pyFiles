# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error, make_scorer
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Label encode 'Sex'
le_sex = LabelEncoder()
df_train['Sex'] = le_sex.fit_transform(df_train['Sex'])
df_test['Sex'] = le_sex.transform(df_test['Sex'])

# RMSLE scorer
def rmsle(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    preds = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, preds))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Feature engineering
df_train['BMI'] = df_train['Weight'] / (df_train['Height']/100)**2
df_test['BMI'] = df_test['Weight'] / (df_test['Height']/100)**2
df_train['MET'] = df_train['Duration'] * df_train['Body_Temp'] / df_train['Age']
df_test['MET'] = df_test['Duration'] * df_test['Body_Temp'] / df_test['Age']
df_train['Duration_HeartRate'] = df_train['Duration'] * df_train['Heart_Rate']
df_test['Duration_HeartRate'] = df_test['Duration'] * df_test['Heart_Rate']
df_train['Age_HeartRate'] = df_train['Age'] * df_train['Heart_Rate']
df_test['Age_HeartRate'] = df_test['Age'] * df_test['Heart_Rate']

# Data preparation
target = 'Calories'
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'MET', 'Age_HeartRate']
features = ['Sex'] + numerical_features + ['Duration_HeartRate']
X_train_full = df_train[features]
y_train_full = df_train[target]
X_test_final = df_test[features]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale numerical features
scaler = MinMaxScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
X_test_final[numerical_features] = scaler.transform(X_test_final[numerical_features])

# Log-transform target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# Model definitions
model_classes = {
    'XGBoost': XGBRegressor(eval_metric='rmse', random_state=42),
    'lightgbm': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=False)
}

param_grids = {
    'XGBoost': {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.8, 0.9]
    },
    'lightgbm': {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [31, 63],
        'feature_fraction': [0.8, 0.9]
    },
    'CatBoost': {
        'iterations': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'depth': [3, 5],
        'subsample': [0.8, 0.9]
    }
}

# Train and evaluate models
log_results = []
trained_models = {}
for name, model_class in model_classes.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(
        model_class,
        param_grids[name],
        cv=5,
        scoring=rmsle_scorer,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_log)
    model = grid_search.best_estimator_
    trained_models[name] = model

    # Predictions
    val_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)

    # Metrics
    val_rmsle = rmsle(y_val_log, val_pred_log)
    test_rmsle = rmsle(y_test_log, test_pred_log)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    log_results.append({
        'Model': name,
        'Validation RMSLE': val_rmsle,
        'Test RMSLE': test_rmsle,
        'Feature Importance': importance
    })

# Ensemble: Weighted averaging based on inverse test RMSLE
weights = [1 / result['Test RMSLE'] for result in log_results]
weights = [w / sum(weights) for w in weights]  # Normalize weights
ensemble_test_pred_log = np.average([model.predict(X_test) for model in trained_models.values()], axis=0, weights=weights)
ensemble_test_rmsle = rmsle(y_test_log, ensemble_test_pred_log)
log_results.append({
    'Model': 'Ensemble',
    'Validation RMSLE': None,  
    'Test RMSLE': ensemble_test_rmsle,
    'Feature Importance': None
})

# Display test scores
print("\nTest RMSLE Scores for All Models:")
for result in log_results:
    print(f"{result['Model']}: Test RMSLE = {result['Test RMSLE']:.4f}")
    if result['Feature Importance'] is not None:
        print(" Feature Importance (Top 3):")
        print(result['Feature Importance'].head(3))
    print()

# Select the best model/ensemble based on test RMSLE
best_result = min(log_results, key=lambda x: x['Test RMSLE'])
best_model_name = best_result['Model']
print(f"\nBest Model/Ensemble (Lowest Test RMSLE): {best_model_name}")

# Generate predictions on test.csv
if best_model_name == 'Ensemble':
    test_pred_log = np.average([model.predict(X_test_final) for model in trained_models.values()], axis=0, weights=weights)
else:
    best_model = trained_models[best_model_name]
    test_pred_log = best_model.predict(X_test_final)



test_pred = np.expm1(test_pred_log)



submit = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submit['Calories'] = test_pred



submit.to_csv('predictions.csv', index=False)


