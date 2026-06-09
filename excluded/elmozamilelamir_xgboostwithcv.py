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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# Label encode the target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['Fertilizer Name'])

# Drop unneeded columns
X = train.drop(columns=['id', 'Fertilizer Name'])
X_test = test.drop(columns=['id'])

# Convert object columns to categorical, then to codes
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

# Initialize
num_classes = len(np.unique(y))
oof_preds = np.zeros((X.shape[0], num_classes))
test_preds = np.zeros((X_test.shape[0], num_classes))

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42 + fold
    }

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=50, verbose_eval=100)

    oof_preds[valid_idx] = model.predict(dvalid)
    test_preds += model.predict(dtest) / skf.n_splits

# Generate submission: top-3 predictions
top3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels = [
    " ".join(label_encoder.inverse_transform(row)) for row in top3
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top3_labels
})

submission.to_csv("submission.csv", index=False)

