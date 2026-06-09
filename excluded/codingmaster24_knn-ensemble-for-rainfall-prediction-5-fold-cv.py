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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target = 'rainfall'
X = df_train[features].copy()
y = df_train[target].copy()
X_test = df_test[features].copy()


X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
knn_predictions = np.zeros(len(X_test))
mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"Training Fold {fold+1}...")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # Distance-weighted KNN
    knn.fit(X_train, y_train)
    
    y_val_pred = knn.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mae_scores.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.4f}")
    
    knn_predictions += knn.predict(X_test_scaled) / 5  # Averaging predictions


print(f"Average MAE across folds: {np.mean(mae_scores):.4f}")


plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), mae_scores, marker='o', linestyle='--', color='b', label='MAE per fold')
plt.xlabel('Fold')
plt.ylabel('Mean Absolute Error')
plt.title('KNN 5-Fold Cross-Validation Performance')
plt.legend()
plt.show()


sample_submission['rainfall'] = knn_predictions
sample_submission.to_csv('knn_ensemble_submission.csv', index=False)
print("Submission file saved as 'knn_ensemble_submission.csv'")


sample_submission

