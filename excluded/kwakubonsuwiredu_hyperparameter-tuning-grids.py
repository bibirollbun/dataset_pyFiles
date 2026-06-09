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


!pip install xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Fill missing values
test_df["winddirection"].fillna(test_df["winddirection"].median(), inplace=True)

# Define features and target
features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target = 'rainfall'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Standardize features
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 1, 2],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [1, 2, 5]
} 

grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
grid_search.fit(X_scaled, y)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train final model with best parameters using K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
final_predictions = np.zeros(len(X_test_scaled))

for train_index, val_index in kf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train XGBoost model with best hyperparameters
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Validate the model
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    auc_scores.append(auc)

    # Predict on test data
    final_predictions += model.predict_proba(X_test_scaled)[:, 1] / kf.n_splits

# Average AUC Score across folds
mean_auc = np.mean(auc_scores)
print(f"Mean AUC-ROC Score: {mean_auc:.5f}")

# Prepare the final submission
test_df["rainfall"] = final_predictions
submission = test_df[["id", "rainfall"]]
submission.to_csv("submission.csv", index=False)

