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
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Initialize and apply LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])  # Male=1, Female=0
test['Sex'] = le.transform(test['Sex'])

train


import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error

# Define features and target
X = train.drop(columns=['id', 'Calories'])
y = train['Calories']
X_test = test.drop(columns=['id'])

# Columns to scale
standardize_cols = ['Height', 'Heart_Rate', 'Weight', 'Body_Temp']
normalize_cols = ['Age', 'Duration']

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), standardize_cols),
        ('norm', MinMaxScaler(), normalize_cols)
    ],
    remainder='passthrough'  # Keep all other columns like 'Sex'
)

# Custom RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Full pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Hyperparameter ranges
param_dist = {
    'model__n_estimators': [700],
    'model__learning_rate': [0.05],
    'model__max_depth': [10,],
    'model__subsample': [ 1],
    'model__colsample_bytree': [0.65],
    'model__gamma': [0.45],
    'model__min_child_weight': [0.5],
    'model__reg_alpha': [0.04],
    'model__reg_lambda': [10]
}
# Random search
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring=rmsle_scorer,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit on training data
search.fit(X, y)

# Evaluate on train
train_preds = np.maximum(search.predict(X), 0)
train_rmsle = rmsle(y, train_preds)
print("Best Parameters:", search.best_params_)
print(f"Train RMSLE: {train_rmsle:.5f}")

# Predict on test and save to submission file
test_preds = np.maximum(search.predict(X_test), 0)
submission = pd.DataFrame({'id': test['id'], 'Calories': test_preds})
submission.to_csv('submission.csv', index=False)


