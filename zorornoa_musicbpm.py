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

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


TARGET = "BeatsPerMinute"
ID = "id"

X = train.drop(columns=[TARGET, ID])
y = train[TARGET]
X_test = test.drop(columns=[ID])


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rf = RandomForestRegressor(random_state=42)


param_grid = {
    "n_estimators": [50, 100, 150], # Reduced number of estimators
    "max_depth": [None, 10, 15], # Limited max_depth
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}


search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=5, # Reduced number of iterations
    scoring="neg_root_mean_squared_error",
    cv=3, # Reduced number of cross-validation folds
    verbose=2,
    random_state=42,
    return_train_score=True,
    n_jobs=-1 # Use all available cores
)


search.fit(X, y)


print("Best Parameters:", search.best_params_)
print("Best CV Score (RMSE):", -search.best_score_)


cv_scores = cross_val_score(
    search.best_estimator_,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)


print("Cross-validated RMSE scores:", -cv_scores)
print("Mean CV RMSE:", -np.mean(cv_scores))


preds = search.best_estimator_.predict(X_test)


submission = pd.DataFrame({
    "Id": np.arange(1, len(preds) + 1),  # adjust if Kaggle provides an Id column
    "Prediction": preds
})


submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




