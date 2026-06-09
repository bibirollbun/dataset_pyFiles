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
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Load dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Define features and target
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target = 'rainfall'
X = df_train[features].copy()
y = df_train[target].copy()
X_test = df_test[features].copy()


# Handle missing values (fill with median)
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())


# Apply Polynomial Features (degree=2 for interactions)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
X_test_scaled = scaler.transform(X_test_poly)


# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_neighbors_values = [3, 5, 9]
knn_predictions = np.zeros(len(X_test))
mae_scores = {n: [] for n in n_neighbors_values}

for n_neighbors in n_neighbors_values:
    print(f"Training KNN with {n_neighbors} neighbors...")
    fold_predictions = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"Fold {fold+1} with {n_neighbors} neighbors...")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        bagging = BaggingRegressor(knn, n_estimators=15, random_state=42, n_jobs=-1)
        bagging.fit(X_train, y_train)
        
        y_val_pred = bagging.predict(X_val)
        mae = mean_absolute_error(y_val, y_val_pred)
        mae_scores[n_neighbors].append(mae)
        print(f"Fold {fold+1} MAE: {mae:.4f}")
        
        fold_predictions += bagging.predict(X_test_scaled) / 5  
    
    knn_predictions += fold_predictions / len(n_neighbors_values)


# Print overall performance
for n_neighbors, scores in mae_scores.items():
    print(f"Average MAE across folds for {n_neighbors} neighbors: {np.mean(scores):.4f}")


# Plot MAE across folds
plt.figure(figsize=(8, 4))
for n_neighbors, scores in mae_scores.items():
    plt.plot(range(1, 6), scores, marker='o', linestyle='--', label=f'K={n_neighbors}')
plt.xlabel('Fold')
plt.ylabel('Mean Absolute Error')
plt.title('KNN Bagging with Polynomial Features - 5-Fold Cross-Validation Performance')
plt.legend()
plt.show()


# Create submission
sample_submission['rainfall'] = knn_predictions
sample_submission.to_csv('knn_bagging_poly_submission.csv', index=False)
print("Submission file saved as 'knn_bagging_poly_submission.csv'")


sample_submission

