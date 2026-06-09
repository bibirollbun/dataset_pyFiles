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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")  # Update with correct path
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")  # Update with correct path

# Define Features and Target
X = train.drop(columns=["id", "rainfall"])  # Drop ID and target column
y = train["rainfall"]  # Target variable
X_test = test.drop(columns=["id"])  # Drop ID from test set

# Normalize Features (optional but helps performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to LightGBM Dataset
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)

# LightGBM Parameters
params = {
    "objective": "regression",  # Predict rainfall probability
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42
}

# Train Model with Correct Syntax
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,  # Maximum boosting rounds
    valid_sets=[lgb_valid],  # Validation dataset
    
)

# Check Best Iteration
print("Best iteration:", model.best_iteration)

# Predict on Validation Set
y_pred_valid = model.predict(X_valid)

# Convert predictions to probabilities using Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_pred_prob = sigmoid(y_pred_valid)

# Evaluate using AUC-ROC
auc_score = roc_auc_score(y_valid, y_pred_prob)
print(f"Validation AUC-ROC Score: {auc_score:.4f}")

# Predict on Test Set
y_test_pred = model.predict(X_test_scaled)
y_test_prob = sigmoid(y_test_pred)

# Create Submission File
submission = pd.DataFrame({"id": test["id"], "rainfall": y_test_prob})
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")



print(train.columns)
print(test.columns)


import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid  # Sigmoid for probability output



# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")  # Update with correct path
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")  # Update with correct path

# Print available columns
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Ensure target column exists in train dataset
target_col = "rainfall"
if target_col not in train.columns:
    raise ValueError(f"Target column '{target_col}' not found in train dataset!")



def feature_engineering(df):
    # Convert 'day' into cyclical seasonal features
    df["sin_day"] = np.sin(2 * np.pi * df["day"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day"] / 365)
    
    # Interaction features
    df["humidity_temp"] = df["humidity"] * df["temparature"]
    df["pressure_wind"] = df["pressure"] * df["windspeed"]
    
    return df

# Apply feature engineering
train = feature_engineering(train)
test = feature_engineering(test)



# Separate target variable
X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test = test.drop(columns=["id"])

# Split into train & validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        
       
    )
    
    y_pred = model.predict(X_valid)
    return roc_auc_score(y_valid, sigmoid(y_pred))

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Get best parameters
best_params = study.best_params
print("Best Parameters:", best_params)



import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from scipy.special import expit as sigmoid  # Convert raw outputs to probability

# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")  # Update with correct path
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")  # Update with correct path

# Feature Engineering
def feature_engineering(df):
    df["sin_day"] = np.sin(2 * np.pi * df["day"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day"] / 365)
    df["humidity_temp"] = df["humidity"] * df["temparature"]
    df["pressure_wind"] = df["pressure"] * df["windspeed"]
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Prepare Data
X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test = test.drop(columns=["id"])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ **Correct way to train LightGBM**
model_lgb = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=50,
    max_depth=10,
    n_estimators=1000  # More trees for better learning
)

# ✅ **Use eval_set & early_stopping_rounds properly**
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[early_stopping(50), log_evaluation(100)]
)

# Predict
y_pred_valid = model_lgb.predict(X_valid)
print("LightGBM Validation AUC-ROC:", roc_auc_score(y_valid, sigmoid(y_pred_valid)))

# Stacking Model
model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
model_ridge = Ridge(alpha=1.0)

stacking_model = StackingRegressor(
    estimators=[
        ('lgbm', model_lgb),
        ('xgb', model_xgb),
        ('ridge', model_ridge)
    ]
)

stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_valid)
print("Stacked Model AUC-ROC:", roc_auc_score(y_valid, sigmoid(y_pred_stacking)))

# Make predictions
y_test_pred = stacking_model.predict(X_test)
y_test_pred = sigmoid(y_test_pred)

# Save submission
submission = pd.DataFrame({"id": test["id"], "rainfall": y_test_pred})
submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved successfully!")



from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation


