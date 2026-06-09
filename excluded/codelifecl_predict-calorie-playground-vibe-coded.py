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


# 1. Imports
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# 2. Load datasets
path = "/kaggle/input/playground-series-s5e5/"
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

# 3. Encode and feature engineer
for df in [train, test]:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2  # Body Mass Index
    df['Effort'] = df['Heart_Rate'] * df['Duration']     # Effort indicator

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Effort']
X = train[features]
y = train['Calories']
X_test = test[features]

# 4. K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))
val_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)
    
    val_pred = np.maximum(0, model.predict(X_val))
    score = np.sqrt(mean_squared_log_error(y_val, val_pred))
    val_scores.append(score)

    test_preds += np.maximum(0, model.predict(X_test)) / kf.n_splits

# 5. Print average CV RMSLE
print(f"CV RMSLE: {np.mean(val_scores):.5f}")

# 6. Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()


submission.head()

