# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#from sklearn.preprocessing import TargetEncoder
from category_encoders.target_encoder import TargetEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
train = pd.concat([train, train_extra], axis=0, ignore_index=True)

test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

target = "Price"

features = [col for col in train.columns if col != target]
CATS = [col for col in train.columns if col not in ["Price", "Weight Capacity (kg)"]]


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import mean_squared_error

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

xgb_params = {
    "device": "cuda",
    "max_depth": 3,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "min_child_weight": 80,
    "enable_categorical": True,
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"### Fold {fold+1} ###")

    X_train, y_train = train.loc[train_idx, features].copy(), train.loc[train_idx, target]
    X_valid, y_valid = train.loc[valid_idx, features].copy(), train.loc[valid_idx, target]
    X_test = test[features].copy()

    TE = TargetEncoder(smoothing=20)
    for col in features:
        TE.fit(X_train[col], y_train)
        X_train[f"TE_{col}"] = TE.transform(X_train[col])
        X_valid[f"TE_{col}"] = TE.transform(X_valid[col])
        X_test[f"TE_{col}"] = TE.transform(X_test[col])

    X_train[CATS] = X_train[CATS].fillna('Missing').astype('category')
    X_valid[CATS] = X_valid[CATS].fillna('Missing').astype('category')
    X_test[CATS] = X_test[CATS].fillna('Missing').astype('category')

    all_features = features + [f"TE_{col}" for col in features]

    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train[all_features], y_train,
        eval_set=[(X_valid[all_features], y_valid)],
        verbose=500
    )

    oof_preds[valid_idx] = model.predict(X_valid[all_features])
    test_preds += model.predict(X_test[all_features]) / FOLDS

rmse = np.sqrt(mean_squared_error(train[target], oof_preds))
print(f"Validation RMSE: {rmse}")



sub = pd.DataFrame({"id": test.index, "Price": test_preds})
sub.to_csv("submission.csv", index=False)
sub.head()

