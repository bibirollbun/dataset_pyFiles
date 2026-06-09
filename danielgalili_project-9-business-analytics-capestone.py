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
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
# import ace_tools as tools

import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge


# Load the data
train = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/train.csv")
test = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/test.csv")
sample_submission = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/sample_submission.csv")


train.head()


test.head()


train.describe()


# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())



# Visualizing the distribution of the target variable (SALE_PRC)
plt.figure(figsize=(8, 5))
sns.histplot(train['SALE_PRC'], bins=50, kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
plt.show()


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Scatter plot of Sale Price vs Living Area
plt.figure(figsize=(8, 5))
sns.scatterplot(x=train['TOT_LVG_AREA'], y=train['SALE_PRC'])
plt.title('Sale Price vs Total Living Area')
plt.xlabel('Total Living Area (sqft)')
plt.ylabel('Sale Price ($)')
plt.show()


# Define numerical and categorical features
num_features = ['TOT_LVG_AREA', 'LND_SQFOOT', 'age', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST']
cat_features = ['structure_quality', 'month_sold', 'avno60plus']

# Target variable
X_train = train.drop(columns=['SALE_PRC', 'id'])
y_train = train['SALE_PRC']
X_test = test.drop(columns=['id'])
test_ids = test['id']

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])


# Random Forest Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)
print(f'Mean Absolute Error (MAE): {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Random Forest Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)
print(f'Mean Absolute Error (MAE): {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Random Forest Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))
])

# Perform 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
mean_cv_mae = -cross_val_mae.mean()
print(f'5-Fold Cross-Validation MAE: {mean_cv_mae}')

# Train the model on full training data
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE on training data
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)
print(f'Mean Absolute Error (MAE) on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Random Forest Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ExtraTreesRegressor(n_estimators=300, random_state=42))
])

# Perform 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
mean_cv_mae = -cross_val_mae.mean()
print(f'5-Fold Cross-Validation MAE: {mean_cv_mae}')

# Train the model on full training data
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE on training data
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)
print(f'Mean Absolute Error (MAE) on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Random Forest Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ExtraTreesRegressor(n_estimators=300, random_state=42))
])

# Perform 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
mean_cv_mae = -cross_val_mae.mean()
print(f'5-Fold Cross-Validation MAE: {mean_cv_mae}')

# Train the model on full training data
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE on training data
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)
print(f'Mean Absolute Error (MAE) on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Define objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 50, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20, step=2)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10, step=1)
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        ))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    mean_cv_mae = -cross_val_mae.mean()
    
    return mean_cv_mae

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, n_jobs=-1)

# Best hyperparameters
best_params = study.best_params

# Train the optimized model
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ExtraTreesRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        n_jobs=-1,
        random_state=42
    ))
])

# Perform cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
mean_cv_mae = -cross_val_mae.mean()

# Train the best model on full training data
best_model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = best_model.predict(X_test)

# Evaluate the best model using MAE on training data
train_pred = best_model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)

# Output results
print(f'Best Parameters: {best_params}')
print(f'Optimized 5-Fold Cross-Validation MAE: {mean_cv_mae}')
print(f'Optimized Mean Absolute Error (MAE) on Training Data: {mae}')



# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission.csv', index=False)


# Define numerical and categorical features
num_features = ['TOT_LVG_AREA', 'LND_SQFOOT', 'age', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST']
cat_features = ['structure_quality', 'month_sold', 'avno60plus']

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

# Define objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 15, step=2)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        ))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    mean_cv_mae = -cross_val_mae.mean()
    
    return mean_cv_mae

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, n_jobs=-1)

# Best hyperparameters
best_params = study.best_params

# Train the optimized model
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    ))
])

# Perform cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
mean_cv_mae = -cross_val_mae.mean()

# Train the best model on full training data
best_model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = best_model.predict(X_test)

# Evaluate the best model using MAE on training data
train_pred = best_model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)

# Output results
print(f'Best Parameters: {best_params}')
print(f'Optimized 5-Fold Cross-Validation MAE: {mean_cv_mae}')
print(f'Optimized Mean Absolute Error (MAE) on Training Data: {mae}')



# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission_xgb.csv', index=False)


# # Define objective function for Optuna
# def objective(trial):
#     n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
#     max_depth = trial.suggest_int('max_depth', 3, 15, step=2)
#     learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    
#     model = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', CatBoostRegressor(
#             iterations=n_estimators,
#             depth=max_depth,
#             learning_rate=learning_rate,
#             loss_function='MAE',
#             random_state=42,
#             verbose=0
#         ))
#     ])

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
#     mean_cv_mae = -cross_val_mae.mean()
    
#     return mean_cv_mae

# # Optimize hyperparameters using Optuna
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# # Best hyperparameters
# best_params = study.best_params

# # Train the optimized model
# best_model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', CatBoostRegressor(
#         iterations=best_params['n_estimators'],
#         depth=best_params['max_depth'],
#         learning_rate=best_params['learning_rate'],
#         loss_function='MAE',
#         random_state=42,
#         verbose=0
#     ))
# ])

# # Perform cross-validation with the best model
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cross_val_mae = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
# mean_cv_mae = -cross_val_mae.mean()

# # Train the best model on full training data
# best_model.fit(X_train, y_train)

# # Make predictions on test dataset
# y_test_pred = best_model.predict(X_test)

# # Evaluate the best model using MAE on training data
# train_pred = best_model.predict(X_train)
# mae = mean_absolute_error(y_train, train_pred)

# # Output results
# print(f'Best Parameters: {best_params}')
# print(f'Optimized 5-Fold Cross-Validation MAE: {mean_cv_mae}')
# print(f'Optimized Mean Absolute Error (MAE) on Training Data: {mae}')


# Define and preprocess features
num_features = ['TOT_LVG_AREA', 'LND_SQFOOT', 'age', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST']
cat_features = ['structure_quality', 'month_sold', 'avno60plus']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

# Train simple CatBoost model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function='MAE',
        random_state=42,
        verbose=0
    ))
])

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
mean_cv_mae = -cross_val_mae.mean()

# Train the model on full training data
model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = model.predict(X_test)

# Evaluate the model using MAE on training data
train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)

# Output results
print(f'5-Fold Cross-Validation MAE: {mean_cv_mae}')
print(f'Mean Absolute Error (MAE) on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission_cat.csv', index=False)


# Define and preprocess features
num_features = ['TOT_LVG_AREA', 'LND_SQFOOT', 'age', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST']
cat_features = ['structure_quality', 'month_sold', 'avno60plus']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

# Train a basic XGBoost model to extract feature importance
initial_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    ))
])

initial_model.fit(X_train, y_train)

# Extract feature importance
feature_importance = initial_model.named_steps['regressor'].feature_importances_
feature_names = num_features + list(initial_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top 10 most important features
top_features = feature_importance_df['Feature'].head(10).tolist()

print("Selected Top Features for Optuna Tuning:")
print(top_features)


# Define Optuna objective function using only important features
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 3, 15, step=2)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    model = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', 'passthrough', [f for f in num_features if f in top_features]),
            ('cat', OneHotEncoder(drop='first'), [f for f in cat_features if f in top_features])
        ])),
        ('regressor', XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=42
        ))
    ])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    mean_cv_mae = -cross_val_mae.mean()
    
    return mean_cv_mae

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, n_jobs=-1)

# Best hyperparameters
best_params = study.best_params

# Train the optimized model using top features
best_model = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', 'passthrough', [f for f in num_features if f in top_features]),
        ('cat', OneHotEncoder(drop='first'), [f for f in cat_features if f in top_features])
    ])),
    ('regressor', XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        objective='reg:squarederror',
        random_state=42
    ))
])

# Perform cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mae = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
mean_cv_mae = -cross_val_mae.mean()

# Train the best model on full training data
best_model.fit(X_train, y_train)

# Make predictions on test dataset
y_test_pred = best_model.predict(X_test)

# Evaluate the best model using MAE on training data
train_pred = best_model.predict(X_train)
mae = mean_absolute_error(y_train, train_pred)

# Output results
print(f'Best Parameters: {best_params}')
print(f'Optimized 5-Fold Cross-Validation MAE: {mean_cv_mae}')
print(f'Optimized Mean Absolute Error (MAE) on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission_xgb__1.csv', index=False)


import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

# Define Optuna objective function
def objective(trial):
    model = lgb.LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        num_leaves=trial.suggest_int('num_leaves', 20, 150, step=10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()  # Minimize MAE

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, n_jobs=-1)

# Train the best model
best_model = lgb.LGBMRegressor(**study.best_params, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mean_cv_mae = -cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1).mean()

best_model.fit(X_train, y_train, categorical_feature=cat_features)
y_test_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_train, best_model.predict(X_train))

# Output results
print(f'Best Parameters: {study.best_params}')
print(f'Optimized 5-Fold Cross-Validation MAE: {mean_cv_mae}')
print(f'Optimized MAE on Training Data: {mae}')


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred})
submission.to_csv('submission_lightgbm.csv', index=False)


# Reduce trials for faster optimization
NUM_TRIALS = 20  

# Normalize numeric features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Define Optuna objective function for LightGBM
def objective_lightgbm(trial):
    model = lgb.LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        num_leaves=trial.suggest_int('num_leaves', 20, 100, step=10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0),
        bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 1.0),
        objective='mae',
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()  # Minimize MAE

# Optimize LightGBM hyperparameters
study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study_lgbm.optimize(objective_lightgbm, n_trials=NUM_TRIALS, n_jobs=-1)

# Train the best LightGBM model
best_lgbm = lgb.LGBMRegressor(**study_lgbm.best_params, objective='mae', random_state=42)
best_lgbm.fit(X_train, y_train, categorical_feature=cat_features)

# Compute Permutation Importance
perm_importance = permutation_importance(best_lgbm, X_train, y_train, scoring='neg_mean_absolute_error', n_repeats=5, random_state=42)
perm_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': perm_importance.importances_mean})
perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)


# Define Optuna objective function for XGBoost
def objective_xgboost(trial):
    model = xgb.XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        objective='reg:squarederror',
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()

# Optimize XGBoost hyperparameters
study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study_xgb.optimize(objective_xgboost, n_trials=NUM_TRIALS, n_jobs=-1)

# Train the best XGBoost model
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_train)

# Combine LightGBM and XGBoost predictions using averaging
y_test_pred_lgbm = best_lgbm.predict(X_test)
y_test_pred_xgb = best_xgb.predict(X_test)
y_test_pred_ensemble = (y_test_pred_lgbm + y_test_pred_xgb) / 2

# Evaluate the ensemble model
train_pred_lgbm = best_lgbm.predict(X_train)
train_pred_xgb = best_xgb.predict(X_train)
train_pred_ensemble = (train_pred_lgbm + train_pred_xgb) / 2

mae_ensemble = mean_absolute_error(y_train, train_pred_ensemble)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Permutation Feature Importance", dataframe=perm_importance_df)

print(f'Best LightGBM Parameters: {study_lgbm.best_params}')
print(f'Best XGBoost Parameters: {study_xgb.best_params}')
print(f'Optimized MAE on Training Data (Ensemble): {mae_ensemble}')



# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred_ensemble})

# Save to CSV
submission.to_csv('submission_ensemble.csv', index=False)

print("Submission file 'submission_ensemble.csv' created successfully!")



# Reduce trials for faster optimization
NUM_TRIALS = 20  

# Normalize numeric features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Define Optuna objective function for CatBoost

def objective_catboost(trial):
    model = cb.CatBoostRegressor(
        iterations=trial.suggest_int('iterations', 100, 1000, step=100),
        depth=trial.suggest_int('depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
        random_state=42,
        verbose=0
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()

# Optimize CatBoost hyperparameters
study_cb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study_cb.optimize(objective_catboost, n_trials=NUM_TRIALS, n_jobs=-1)

# Train the best CatBoost model
best_cb = cb.CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_cb.fit(X_train, y_train, cat_features=cat_features)


# Reduce trials for faster optimization
NUM_TRIALS = 20  

# Normalize numeric features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Define Optuna objective function for CatBoost

def objective_catboost(trial):
    model = cb.CatBoostRegressor(
        iterations=trial.suggest_int('iterations', 100, 1000, step=100),
        depth=trial.suggest_int('depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
        random_state=42,
        verbose=0
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()

# Optimize CatBoost hyperparameters
study_cb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study_cb.optimize(objective_catboost, n_trials=NUM_TRIALS, n_jobs=-1)

# Train the best CatBoost model
best_cb = cb.CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_cb.fit(X_train, y_train, cat_features=cat_features)

# Define Optuna objective function for XGBoost

def objective_xgboost(trial):
    model = xgb.XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        objective='reg:squarederror',
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_mae = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
    
    return -cross_val_mae.mean()

# Optimize XGBoost hyperparameters
study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study_xgb.optimize(objective_xgboost, n_trials=NUM_TRIALS, n_jobs=-1)

# Train the best XGBoost model
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_train)

# Compute Permutation Importance for CatBoost
perm_importance_cb = permutation_importance(best_cb, X_train, y_train, scoring='neg_mean_absolute_error', n_repeats=5, random_state=42)
perm_importance_df_cb = pd.DataFrame({'Feature': X_train.columns, 'Importance': perm_importance_cb.importances_mean})
perm_importance_df_cb = perm_importance_df_cb.sort_values(by='Importance', ascending=False)

# Combine CatBoost and XGBoost predictions using averaging
y_test_pred_cb = best_cb.predict(X_test)
y_test_pred_xgb = best_xgb.predict(X_test)
y_test_pred_ensemble = (y_test_pred_cb + y_test_pred_xgb) / 2

# Evaluate the ensemble model
train_pred_cb = best_cb.predict(X_train)
train_pred_xgb = best_xgb.predict(X_train)
train_pred_ensemble = (train_pred_cb + train_pred_xgb) / 2

mae_ensemble = mean_absolute_error(y_train, train_pred_ensemble)

# Display results
print(perm_importance_df_cb.head())
perm_importance_df_cb.plot(kind='bar', x='Feature', y='Importance', title='Permutation Feature Importance')
plt.show()

print(f'Best CatBoost Parameters: {study_cb.best_params}')
print(f'Best XGBoost Parameters: {study_xgb.best_params}')
print(f'Optimized MAE on Training Data (Ensemble): {mae_ensemble}')


# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_test_pred_ensemble})

# Save to CSV
submission.to_csv('submission_ensemble_1.csv', index=False)

print("Submission file 'submission_ensemble.csv' created successfully!")


# Reduce trials for faster optimization
NUM_TRIALS = 20  

# Normalize numeric features
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# Define Optuna objective function for CatBoost
def objective_catboost(trial):
    model = CatBoostRegressor(
        iterations=trial.suggest_int('iterations', 100, 1000, step=100),
        depth=trial.suggest_int('depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
        random_state=42,
        verbose=0
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1).mean()

study_cb = optuna.create_study(direction='minimize')
study_cb.optimize(objective_catboost, n_trials=20, n_jobs=-1)
best_cb = CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_cb.fit(X_train, y_train)

# Define Optuna objective function for XGBoost
def objective_xgboost(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        objective='reg:squarederror',
        random_state=42
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1).mean()

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgboost, n_trials=20, n_jobs=-1)
best_xgb = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_train)

# Train Random Forest
def objective_rf(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
        random_state=42, n_jobs=-1
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1).mean()

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=20, n_jobs=-1)
best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)



# Evaluate the ensemble model
train_pred_cb = best_cb.predict(X_train)
train_pred_xgb = best_xgb.predict(X_train)
train_pred_ensemble = (train_pred_cb + train_pred_xgb) / 2

mae_ensemble = mean_absolute_error(y_train, train_pred_ensemble)



# # Create submission DataFrame
# submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_pred_ensemble})

# # Save to CSV
# submission.to_csv('submission_ensemble_2.csv', index=False)

# print("Submission file 'submission_ensemble.csv' created successfully!")


# Train the optimized base models
best_cb = CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_xgb = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42)
best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)

# Define the Stacking Regressor
stacking_reg = StackingRegressor(
    estimators=[
        ('catboost', best_cb),
        ('xgboost', best_xgb),
        ('random_forest', best_rf)
    ],
    final_estimator=Ridge(alpha=1.0),  # Meta-model
    n_jobs=-1
)

# Fit the stacking model
stacking_reg.fit(X_train, y_train)

# Predict on training set and evaluate
train_pred_stacked = stacking_reg.predict(X_train)
mae_stacked = mean_absolute_error(y_train, train_pred_stacked)
print(f"Stacking Model MAE: {mae_stacked}")

# Make final predictions
y_pred_stacked = stacking_reg.predict(X_test)


# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_pred_stacked})

# Save to CSV
submission.to_csv('submission_stacking.csv', index=False)

print("Submission file 'submission_stacking.csv' created successfully!")


# Train the optimized base models
best_cb = CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_xgb = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42)
# best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)

# Define the Stacking Regressor
stacking_reg = StackingRegressor(
    estimators=[
        ('catboost', best_cb),
        ('xgboost', best_xgb)
    ],
    final_estimator=Ridge(alpha=1.0),  # Meta-model
    n_jobs=-1
)

# Fit the stacking model
stacking_reg.fit(X_train, y_train)

# Predict on training set and evaluate
train_pred_stacked = stacking_reg.predict(X_train)
mae_stacked = mean_absolute_error(y_train, train_pred_stacked)
print(f"Stacking Model MAE: {mae_stacked}")

# Make final predictions
y_pred_stacked = stacking_reg.predict(X_test)


# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'SALE_PRC': y_pred_stacked})

# Save to CSV
submission.to_csv('submission_stacking_1.csv', index=False)

print("Submission file 'submission_stacking_1.csv' created successfully!")




