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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score
from scipy.stats import uniform, randint



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
train.head(5)


train.describe().T



num_features = train.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("id")
plt.figure(figsize=(10,8))
sns.heatmap(train[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



train[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


# Diagnostic: Print distribution of beatsperminute in training set
print('beatsperminute distribution in training set:')
print('Min:', train['BeatsPerMinute'].min())
print('Max:', train['BeatsPerMinute'].max())
print('Mean:', train['BeatsPerMinute'].mean())
print('Std:', train['BeatsPerMinute'].std())


train = train.drop('id', axis=1)

test = test.drop('id', axis=1)



X = train.drop(['BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']


X = pd.get_dummies(X)
test = pd.get_dummies(test)

X, test = X.align(test, join='left', axis=1, fill_value=0)


# Feature engineering: train set
X['Energy_x_Acoustic'] = X['Energy'] * X['AcousticQuality']
X['Loud_norm_Energy']  = X['AudioLoudness'] / (X['Energy'] + 1e-5)
X['Vocal_x_Live']      = X['VocalContent'] * X['LivePerformanceLikelihood']
X['TrackDurationMin'] = X['TrackDurationMs'] / 60000
# Feature engineering: test set
test['Energy_x_Acoustic'] = test['Energy'] * test['AcousticQuality']
test['Loud_norm_Energy']  = test['AudioLoudness'] / (test['Energy'] + 1e-5)
test['Vocal_x_Live']      = test['VocalContent'] * test['LivePerformanceLikelihood']
test['TrackDurationMin'] = test['TrackDurationMs'] / 60000


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.19),  # uniform from 0.01 to 0.2
    'subsample': uniform(0.7, 0.3),        # uniform from 0.7 to 1.0
    'colsample_bytree': uniform(0.7, 0.3), # uniform from 0.7 to 1.0
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 1),
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0, 2)
}

# Create base model
base_model = XGBRegressor(random_state=42)

# Setup RandomizedSearchCV
print("\nStarting RandomizedSearchCV to find best parameters...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=8,                           # Number of parameter settings sampled
    cv=3,                               # 5-fold cross-validation
    scoring=make_scorer(rmse, greater_is_better=False),  # Custom RMSE scorer
    random_state=42,
    n_jobs=-1,                          # Use all available cores
    verbose=2                           # Show progress
)

# Fit RandomizedSearchCV
random_search.fit(X, y)

# Print best parameters and scores
print("\nBest parameters found by RandomizedSearchCV:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"Best RMSE: {-random_search.best_score_:.4f}")



best_params = random_search.best_params_

model = XGBRegressor(**best_params)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model.fit(X, y)



importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances (XGBoost)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feat_names[indices], rotation=90)
plt.tight_layout()
plt.show()


submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
submission['BeatsPerMinute'] = model.predict(test)
submission.to_csv('submission.csv', index=False)


predictions = model.predict(test)
print("\nPrediction statistics:")
print(f"Min: {predictions.min():.2f}")
print(f"Max: {predictions.max():.2f}")
print(f"Mean: {predictions.mean():.2f}")
print(f"Std: {predictions.std():.2f}")




