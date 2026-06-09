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
from catboost import CatBoostClassifier, Pool

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


## THIS PART IS FROM KaanNakipoglu 
cat_cols = [c for c in train.columns
            if train[c].dtype == "object" and c not in (TARGET, ID_COL)]

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col]  = test[col].astype("category")

label_map = {lbl: i for i, lbl in enumerate(sorted(train[TARGET].unique()))}
inv_map   = {i: lbl for lbl, i in label_map.items()}
train["label_int"] = train[TARGET].map(label_map)
test["label_int"] = train[TARGET].map(label_map)

features = [c for c in train.columns if c not in (TARGET, ID_COL)] #, "label_int"


print(features)


train.head(10)


test.head(10)


# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['Fertilizer_Name'])

# Drop ID and target from features
X = train.drop(columns=['id', 'Fertilizer_Name'])
X_test = test.drop(columns=['id'])

# Identify categorical features (CatBoost handles them natively)
#cat_cols = X.select_dtypes(include='object').columns.tolist()

# Prepare test prediction array
num_classes = len(np.unique(y))
test_preds = np.zeros((X_test.shape[0], num_classes))


# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=100,
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

    test_preds += model.predict_proba(X_test) / skf.n_splits



# Format top-3 predictions
#top3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
#top3_labels = [
#    " ".join(label_encoder.inverse_transform(row)) for row in top3
#]

top3_idx  = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]

submission_labels = (
    pd.DataFrame(top3_idx)
      .replace(inv_map)
      .agg(" ".join, axis=1)
)


submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": submission_labels
})
submission.to_csv("submission.csv", index=False)


submission.head(10)

