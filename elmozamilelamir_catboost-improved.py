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
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

def tidy(df):
    df.columns = (df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("-", "_"))
    return df

train = tidy(train)
test = tidy(test)

TARGET = "Fertilizer_Name"
ID_COL  = "id"

# Identify categorical columns
cat_cols = [col for col in train.columns 
            if train[col].dtype == "object" and col != TARGET]

# Consistent category types
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    train[col] = pd.Categorical(train[col], categories=combined.unique())
    test[col] = pd.Categorical(test[col], categories=combined.unique())

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train[TARGET])
inv_map = dict(enumerate(label_encoder.classes_))

# Drop unnecessary columns
features = [col for col in train.columns if col not in [ID_COL, TARGET]]
X = train[features]
X_test = test[features]

# Prediction array
num_classes = len(label_encoder.classes_)
test_preds = np.zeros((X_test.shape[0], num_classes))


# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        random_strength=1.5,
        border_count=128,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        cat_features=cat_cols,
        random_seed=42 + fold,
        task_type="CPU",  # Switch to GPU if available
        verbose=300,
        early_stopping_rounds=200
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    test_preds += model.predict_proba(X_test) / skf.n_splits


# Format top-3 predictions
top3_idx = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
submission_labels = (
    pd.DataFrame(top3_idx)
      .replace(inv_map)
      .agg(" ".join, axis=1)
)


# Submit
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": submission_labels
})
submission.to_csv("submission.csv", index=False)

