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


#Required modules 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import xgboost as xgb


# Load Data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
ids = test_data["id"]


train_data.head()


train_data.nunique()


train_data.dtypes


# Initial Preprocessing
def preprocess(df):
    df = df.drop(columns=["id", "Weight Capacity (kg)"], errors="ignore")
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

train_data = preprocess(train_data)
test_data = preprocess(test_data)


# One-Hot Encoding
cat_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Waterproof', 'Laptop Compartment']
train_data = pd.get_dummies(train_data, columns=cat_cols)
test_data = pd.get_dummies(test_data, columns=cat_cols)


# Align features between train and test
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)


# Prepare data
X = train_data.drop(columns=["Price"])
y = np.log1p(train_data["Price"])  # Log transformation


# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter Tuning (XGBoost)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, cv=3, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_



# Hyperparameter Tuning (Random Forest)
rf_model = RandomForestRegressor(random_state=42)
rf_params = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=10, cv=3, 
                              scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_



# Stacking Ensemble
stack_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf)
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)



# Train and Evaluate
stack_model.fit(X_train, y_train)
val_pred = stack_model.predict(X_val)
mse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {mse:.4f} (log space)")
print(f"Validation RMSE (original scale): {np.expm1(mse):.4f}")


# Generate Predictions
test_pred = stack_model.predict(test_data)
submission = pd.DataFrame({
    'id': ids,
    'Price': np.expm1(test_pred)  # Reverse log transformation
})


# Save Submission
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")

