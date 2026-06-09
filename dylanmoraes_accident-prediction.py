


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


data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
data.shape


data.head()


# 1. Imports and basic setup
import pandas as pd

# 2. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# 3. Quick look at structure and sample rows
print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())
display(train.info())



# Check for missing values
display(train.isnull().sum())

# Show datatypes for all columns
display(train.dtypes)



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))
sns.histplot(train['accident_risk'], bins=30, kde=True)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()



categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 'holiday', 'school_season', 'public_road', 'road_signs_present']

for col in categorical_cols:
    plt.figure(figsize=(8, 3))
    sns.countplot(x=col, data=train)
    plt.title(f'{col} value counts')
    plt.show()
    
    plt.figure(figsize=(8, 3))
    sns.barplot(x=col, y='accident_risk', data=train, estimator='mean')
    plt.title(f'Mean Accident Risk by {col}')
    plt.show()



num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

for col in num_cols:
    plt.figure(figsize=(8, 3))
    sns.histplot(train[col], bins=30, kde=True)
    plt.title(f'{col} Distribution')
    plt.show()



for col in num_cols:
    plt.figure(figsize=(8, 3))
    # Bin the numerical feature for visualization
    train['binned'] = pd.cut(train[col], bins=10)
    sns.boxplot(x='binned', y='accident_risk', data=train)
    plt.title(f'Accident Risk by {col} (binned)')
    plt.xticks(rotation=45)
    plt.show()
    train.drop('binned', axis=1, inplace=True)



# Focus only on numerical features
corr = train[num_cols + ['accident_risk']].corr()
plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Numerical Feature Correlations')
plt.show()



# Example 1: Road Type vs. Lighting and Mean Accident Risk
plt.figure(figsize=(10, 5))
sns.pointplot(x='road_type', y='accident_risk', hue='lighting', data=train, estimator='mean')
plt.title('Mean Accident Risk: Road Type vs. Lighting')
plt.show()

# Curvature (binned) vs Weather - Updated to handle multiple hue levels
train['curv_bin'] = pd.cut(train['curvature'], bins=5)
plt.figure(figsize=(10, 5))
sns.violinplot(x='curv_bin', y='accident_risk', hue='weather', data=train, split=False)
plt.title('Accident Risk by Curvature (Binned) and Weather')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()
train.drop('curv_bin', axis=1, inplace=True)

# Example 3: Time of Day vs. Accident Risk
plt.figure(figsize=(8, 4))
sns.boxplot(x='time_of_day', y='accident_risk', data=train)
plt.title('Accident Risk by Time of Day')
plt.show()



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Select features (drop id, target)
features = [col for col in train.columns if col not in ['id', 'accident_risk']]
X = train[features]
y = train['accident_risk']

# Detect categorical and numerical features
categorical = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

# Example: Preprocess training data (fit and transform)
X_prep = preprocessor.fit_transform(X)

print("Preprocessed shape:", X_prep.shape)



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Train/validation split for local model check
X_train, X_val, y_train, y_val = train_test_split(X_prep, y, test_size=0.2, random_state=42)

# 2. Fit linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
val_preds = lr.predict(X_val)

# 3. Calculate RMSE
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Linear Regression validation RMSE:", rmse)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_val_preds = rf.predict(X_val)
rf_rmse = mean_squared_error(y_val, rf_val_preds, squared=False)
print("Random Forest validation RMSE:", rf_rmse)



import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_rmse = mean_squared_error(y_val, xgb_val_preds, squared=False)
print("XGBoost validation RMSE:", xgb_rmse)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_val_preds = rf.predict(X_val)
rf_rmse = mean_squared_error(y_val, rf_val_preds, squared=False)
print("Random Forest validation RMSE:", rf_rmse)



from sklearn.model_selection import GridSearchCV

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2
)
rf_gs.fit(X_train, y_train)
print("Best RF params:", rf_gs.best_params_)
print("Best RF RMSE:", -rf_gs.best_score_)



import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_rmse = mean_squared_error(y_val, xgb_val_preds, squared=False)
print("XGBoost validation RMSE:", xgb_rmse)



xgb_param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
xgb_gs = GridSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    xgb_param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2
)
xgb_gs.fit(X_train, y_train)
print("Best XGB params:", xgb_gs.best_params_)
print("Best XGB RMSE:", -xgb_gs.best_score_)



import matplotlib.pyplot as plt

# For Random Forest
importances = rf_gs.best_estimator_.feature_importances_
feature_names = preprocessor.get_feature_names_out()
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.title('Top Feature Importances (Random Forest)')
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()



rf_preds = rf_gs.best_estimator_.predict(X_val)
xgb_preds = xgb_gs.best_estimator_.predict(X_val)
ensemble_preds = 0.5 * rf_preds + 0.5 * xgb_preds

from sklearn.metrics import mean_squared_error
ensemble_rmse = mean_squared_error(y_val, ensemble_preds, squared=False)
print("Ensemble validation RMSE:", ensemble_rmse)



best_model = xgb_gs.best_estimator_    # or rf_gs.best_estimator_, depending on the best RMSE
best_model.fit(X_prep, y)

# Preprocess test set
X_test = test[features]
X_test_prep = preprocessor.transform(X_test)
submission_preds = best_model.predict(X_test_prep)

# Clip preds to [0, 1] as required by competition
submission_preds = submission_preds.clip(0, 1)

# Prepare for submission
submission = pd.DataFrame({'id': test['id'], 'accident_risk': submission_preds})
submission.to_csv('submission.csv', index=False)



# 1. Generate test predictions with your best model (e.g., XGBoost, Random Forest, or Ensemble)
test_features = test[features]
test_prep = preprocessor.transform(test_features)
test_preds = best_model.predict(test_prep)  # replace best_model as appropriate

# 2. Clip predictions to [0, 1] (competition requirement)
test_preds = test_preds.clip(0, 1)

# 3. Prepare submission dataframe
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = test_preds

# 4. Save to CSV for submission
submission.to_csv('submission.csv', index=False)





