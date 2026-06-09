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


# Step 1: Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/test.csv')

# Step 3: Data Preparation
# Separating features and target
X = train_data.drop(columns=['target'])  # Exclude only the target column
y = train_data['target']
X_test = test_data.drop(columns=['id'])  # Exclude the 'id' column from test data

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Training and Evaluation
# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Validation
y_val_pred = rf_model.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R^2 Score: {r2}")

# Hyperparameter tuning (optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Recalculate validation score with the best model
y_val_pred_best = best_model.predict(X_val)
r2_best = r2_score(y_val, y_val_pred_best)
print(f"Validation R^2 Score with Best Model: {r2_best}")

# Step 5: Make predictions on test data
y_test_pred = best_model.predict(X_test_scaled)

# Step 6: Create submission file
submission = pd.DataFrame({'id': test_data['id'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")



# Step 1: Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Step 2: Load the Data
train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/test.csv')

# Step 3: Data Preparation
# Separating features and target
X = train_data.drop(columns=['target'])  # Exclude target column
y = train_data['target']
X_test = test_data.drop(columns=['id'])  # Exclude id column from test data

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train-Test Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Define Models
# XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6
)

# LightGBM
lgb_model = lgb.LGBMRegressor(
    objective='regression', 
    random_state=42, 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6
)

# CatBoost
catboost_model = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.05, 
    depth=6, 
    random_state=42, 
    verbose=0
)

# Step 6: Stacking Regressor
estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('catboost', catboost_model)
]

stacking_model = StackingRegressor(
    estimators=estimators, 
    final_estimator=xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        n_estimators=200
    )
)

# Step 7: Train the Stacking Model
stacking_model.fit(X_train, y_train)

# Validation
y_val_pred = stacking_model.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R^2 Score: {r2}")

# Step 8: Make Predictions on Test Data
y_test_pred = stacking_model.predict(X_test_scaled)

# Step 9: Create Submission File
submission = pd.DataFrame({'id': test_data['id'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")



# Step 1: Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Data
train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/test.csv')

# Step 3: Data Preparation
# Separating features and target
X = train_data.drop(columns=['target'])  # Exclude target column
y = train_data['target']
X_test = test_data.drop(columns=['id'])  # Exclude id column from test data

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train-Test Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Define Models
# ElasticNet (Linear Model)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# HistGradientBoostingRegressor (Tree-based Model)
hgb_regressor = HistGradientBoostingRegressor(
    max_iter=500, 
    max_depth=10, 
    learning_rate=0.05, 
    random_state=42
)

# Step 6: Ensemble (Voting Regressor)
voting_regressor = VotingRegressor(
    estimators=[
        ('elasticnet', elastic_net), 
        ('hgb', hgb_regressor)
    ]
)

# Step 7: Train the Ensemble Model
voting_regressor.fit(X_train, y_train)

# Validation
y_val_pred = voting_regressor.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R^2 Score: {r2}")

# Step 8: Make Predictions on Test Data
y_test_pred = voting_regressor.predict(X_test_scaled)

# Step 9: Create Submission File
submission = pd.DataFrame({'id': test_data['id'], 'target': y_test_pred})
submission.to_csv('submission_voting.csv', index=False)
print("Submission file created successfully.")



# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Step 2: Load Data
train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/archive/test.csv')

# Step 3: Data Preparation
# Separating features and target
X = train_data.drop(columns=['target'])  # Exclude target column
y = train_data['target']
X_test = test_data.drop(columns=['id'])  # Exclude id column from test data

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Define and Train XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Step 6: Validate the Model
y_val_pred = xgb_model.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R^2 Score: {r2}")

# Step 7: Make Predictions on Test Data
y_test_pred = xgb_model.predict(X_test_scaled)

# Step 8: Create Submission File
submission = pd.DataFrame({'id': test_data['id'], 'target': y_test_pred})
submission.to_csv('submission_xgboost.csv', index=False)
print("Submission file created successfully.")





