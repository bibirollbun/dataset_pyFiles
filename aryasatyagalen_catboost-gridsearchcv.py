import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install scikit-learn==1.6.1 xgboost==3.0.2 lightgbm==4.6.0 catboost===1.2.8 optuna==4.5.0


import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import math
from sklearn.model_selection import GridSearchCV



train = pd.DataFrame(pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv"))
test = pd.DataFrame(pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv"))
display(train)
display(test)


cat_cols = train.select_dtypes(include=['object','category']).columns
list_cat = cat_cols.tolist()
print(list_cat)


train_encoded = pd.get_dummies(train, columns=list_cat)
test_encoded = pd.get_dummies(test, columns=list_cat)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
display(test_encoded.columns)
display(train_encoded.columns)


X_train = train_encoded.drop(columns=['price', 'id'])
y_train = train_encoded['price']
X_test = test_encoded.drop(columns=['id'], axis=1)


model = CatBoostRegressor(loss_function='RMSE', eval_metric='R2', verbose=0, random_seed=42)

param_grid = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'iterations': [500, 1000, 2000]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)


best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    "id": test['id'],
    "y": test_preds
})


submission.to_csv('submission_tuned_v1.csv', index=False)


