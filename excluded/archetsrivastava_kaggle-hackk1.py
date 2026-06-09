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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
trainData = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')  
testData = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')


feature_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
X = trainData[feature_columns]
y = trainData['target']
X_test = testData[feature_columns]
scaler=QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.020, random_state=42)


param_dist = {
    'n_estimators': [300, 400, 500, 1000],  # Try increasing n_estimators
    'max_depth': [10, 15, 20, 25, None],  # Add more max_depth options
    'min_samples_split': [2, 5, 10, 20],  # Additional values for min_samples_split
    'min_samples_leaf': [1, 2, 4, 6],  # Experiment with different values for min_samples_leaf
    'max_features': ['sqrt', 'log2', 5, 6],  # Try absolute number of features
    'bootstrap': [True, False]  # Test both bootstrap strategies
}


random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    scoring='r2',
    cv=5,  
    verbose=2,
    n_jobs=-1,
    n_iter=40,
)


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
val_predictions = best_model.predict(X_val)
val_r2 = r2_score(y_val, val_predictions)
print(f"Validation R² Score: {val_r2:.4f}")


test_predicted = best_model.predict(X_test_scaled)
submission_df = pd.DataFrame({
     'id': testData['id'],
    'target': test_predicted
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

