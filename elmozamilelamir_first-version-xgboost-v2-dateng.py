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


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("Train: ", train.shape, "   Test:", test.shape)
train.head()


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

features = [c for c in train.columns if c not in (TARGET, "label_int", ID_COL)]


X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], train["label_int"], test_size=0.2, random_state=42, stratify=train["label_int"]
)

print("Categorical columns:", cat_cols)
print("Classes:", label_map)


dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical = 1)
dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical = 1)


params = {
    'objective': 'multi:softprob',
    'num_class': train["label_int"].nunique(),
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, num_boost_round=1000,
                  early_stopping_rounds=50, evals=watchlist, verbose_eval=50)



test_df = xgb.DMatrix(test[features], enable_categorical = 1)


y_pred = model.predict(test_df) 


top3_idx  = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]

submission_labels = (
    pd.DataFrame(top3_idx)
      .replace(inv_map)
      .agg(" ".join, axis=1)
)

sub = sample_sub
SUB_COL = "Fertilizer Name"
sub[SUB_COL] = submission_labels


print("creating submission file")

sub.to_csv("submission.csv", index=False)

