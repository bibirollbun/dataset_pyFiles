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


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



test=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
train=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')



train.head()


train.describe()


print(train.columns)



X_train = train
y_train = X_train.pop('target')
X_test = test.drop(columns=['id']) 



y_train


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



test


# from xgboost import XGBRegressor
# model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
# model.fit(X_train, y_train)


# predictions = model.predict(X_test)



# submission = pd.DataFrame({'id': test['id'], 'target': predictions})
# submission.to_csv('submission3.csv',index=False)


# model = XGBRegressor(
#     n_estimators=500,          # Number of trees
#     learning_rate=0.05,        # Step size shrinkage to prevent overfitting
#     max_depth=8,               # Maximum tree depth
#     subsample=0.8,             # Fraction of samples for each tree
#     colsample_bytree=0.8,      # Fraction of features for tree construction
#     n_jobs=-1,                 # Parallel computation
#     random_state=42            # For reproducibility
# )
# model.fit(
#     X_train_split, y_train_split,
#     eval_set=[(X_valid_split, y_valid_split)],
#     eval_metric='rmse',
#     early_stopping_rounds=50,
#     verbose=50 
# )


# y_valid_pred = model.predict(X_valid_split)
# valid_r2_score = r2_score(y_valid_split, y_valid_pred)
# y_valid_pred
# valid_r2_score


# param_grid = {
#     'n_estimators': [100, 200, 500],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [6, 8, 10],
#     'subsample': [0.8, 0.9],
#     'colsample_bytree': [0.8, 1.0],
#     'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
#     'reg_lambda': [1.0, 2.0],    # L2 regularization
# }
# grid_search = GridSearchCV(
#     estimator=XGBRegressor(random_state=42, n_jobs=-1),
#     param_grid=param_grid,
#     scoring='r2',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )
# grid_search.fit(X_train_split, y_train_split)

# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_

# best_model.fit(X_train, y_train)
# predictions = best_model.predict(X_test)


# submission = pd.DataFrame({'id': test['id'], 'target': predictions})
# submission.to_csv('submission4.csv', index=False)


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
    'reg_lambda': [1.0, 2.0, 3.0]  # L2 regularization
}

xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1, eval_metric="rmse"),
    param_grid=xgb_param_grid,
    cv=5,  
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

xgb_grid.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train)],
    early_stopping_rounds=10, 
    verbose=1
)

xgb_best_model = xgb_grid.best_estimator_
best_params = xgb_grid.best_params_

print("Best Parameters:", best_params)

predictions = xgb_best_model.predict(X_test)

# save_submission(test_ids, xgb_predictions, 'submission_xgb_optimized.csv')
submission = pd.DataFrame({'id': test['id'], 'target': predictions})
submission.to_csv('submission_xgboost.csv', index=False)


