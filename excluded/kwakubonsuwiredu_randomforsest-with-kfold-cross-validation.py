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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
submission_path = "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

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

# Initialize Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
final_predictions = np.zeros(len(X_test_scaled))

# Train model using K-Fold Cross-Validation
for train_index, val_index in kf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
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

