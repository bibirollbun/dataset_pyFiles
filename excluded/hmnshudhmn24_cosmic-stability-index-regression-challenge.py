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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# Load datasets
train = pd.read_csv("/kaggle/input/tda-aiml-cosmic-stability-problem-0f3ebc/train.csv")
test = pd.read_csv("/kaggle/input/tda-aiml-cosmic-stability-problem-0f3ebc/test.csv")

# Split features and target
X = train.drop(columns=["cosmic_stability_index"])
y = train["cosmic_stability_index"]
X_test = test.copy()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    test_predictions += model.predict(X_test) / kf.n_splits

# Create submission file
submission = pd.DataFrame({
    "id": X_test["id"],
    "cosmic_stability_index": test_predictions
})

submission.to_csv("submission.csv", index=False)

submission.head()


