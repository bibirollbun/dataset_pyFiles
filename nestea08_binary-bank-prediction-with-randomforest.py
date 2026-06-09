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

import warnings



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


train.info()


train.isna().sum()


test.isna().sum()


X = train.drop(['id','y'], axis=1)
y = train['y']



test_id = test['id']
test=test.drop('id', axis=1)


from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline








train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, 
                                                    random_state=2)



categorical_cols = [c for c in train_X.columns if train_X[c].dtype == "object"]
numeric_cols = [c for c in train_X.columns if train_X[c].dtype in ['int64', 'float64']]

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])


def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = Pipeline(steps=[('preprocessor', preprocessor),
    ('model', RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=2))
       ])
    model.fit(train_X, train_y)
    preds_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_y)
    return(mae, model)


leaf_nodes_size = [5, 25, 50, 100, 250, 500]
best_tree_size = None
lowest_mae = float("inf")

for leaf_size in leaf_nodes_size:
    mean_err, model = get_mae(leaf_size, train_X, test_X, train_y, test_y)
    preds_y = model.predict(test_X)
    roc_auc = roc_auc_score(test_y, preds_y)
    print(f"Leaf nodes: {leaf_size}\t Mean Absolute Error: {mean_err:.2f}\t ROC: {roc_auc:.4f}")
    
best_tree_size = min(leaf_nodes_size, 
                     key=lambda leaf_size: get_mae(leaf_size, train_X, test_X, train_y, test_y))






test_X.head()


test_predict = model.predict(test)


submission = pd.DataFrame({
    'id': test_id,
    'y': test_predict
})

submission.to_csv('submission.csv', index=False)

