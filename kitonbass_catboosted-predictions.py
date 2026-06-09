import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head()


train.columns


plt.hist(train['BeatsPerMinute'], bins=20)
mean_val = train['BeatsPerMinute'].mean()
std = np.std(train['BeatsPerMinute'])
plt.axvline(mean_val, color='salmon', linestyle='--', linewidth=2)
plt.axvline(mean_val + 2.5 * std, color='salmon', linestyle='--', linewidth=2)
plt.axvline(mean_val - 2.5 * std, color='salmon', linestyle='--', linewidth=2)
plt.title('Histogram and mean')


import seaborn as sns
sns.boxplot(train['BeatsPerMinute'])
plt.title('Boxplot (shows outliers)')


upper_bound = 185
lower_bound = 50

len_before = len(train)
train = train[(train['BeatsPerMinute'] > lower_bound) & (train['BeatsPerMinute'] < upper_bound)]
len_after = len(train)

print(f'{(1 - len_after/len_before) * 100:.2f}% of original data was removed')


sns.boxplot(train['BeatsPerMinute'])
plt.title('Boxplot of clean data')


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from catboost import CatBoostRegressor


X, y = train.drop(['BeatsPerMinute'], axis=1), train['BeatsPerMinute']


X


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)


regressor = make_pipeline(StandardScaler(), SplineTransformer(degree=3, n_knots=3), CatBoostRegressor())
param_grid = { 'catboostregressor__learning_rate': (0.01, 0.05, 0.1),
                'catboostregressor__l2_leaf_reg': (1, 3, 5),
                'catboostregressor__depth': [4, 6, 8],
                'catboostregressor__n_estimators': [100, 1000, 2000],
                'catboostregressor__verbose':[False]
               }
search = GridSearchCV(regressor, param_grid, cv=5, verbose=2, scoring='neg_root_mean_squared_error',refit=True)
search.fit(X_train, y_train)


search.best_score_


search.best_params_


def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))


pred = search.predict(X_val)
rmse(y_val, pred)


pred = pd.DataFrame({
    'id': test.id,
    'BeatsPerMinute': search.predict(test)
})
pred.to_csv('best_catboost_prediction_no_idx.csv', index=False)

