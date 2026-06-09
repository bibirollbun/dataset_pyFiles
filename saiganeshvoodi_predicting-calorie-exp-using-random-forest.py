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
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


train_path = '/kaggle/input/playground-series-s5e5/train.csv'
test_path = '/kaggle/input/playground-series-s5e5/test.csv'
submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'



train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(submission_path)


train.head()


print(train.isnull().sum())
print(test.isnull().sum())


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


print(train.describe())


sns.histplot(train['Calories'], bins=50, kde=True)
plt.title("Calories Distribution")
plt.show()



sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


X = train.drop(columns=['Calories', 'id'])
y = train['Calories']
X_test = test.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)



param_grid = {
    'n_estimators': [100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, scoring=rmsle_scorer, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


val_preds = best_model.predict(X_val)
print(f"Validation RMSLE: {rmsle(y_val, val_preds):.4f}")



test_preds = best_model.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)


submission.isnull().sum()




