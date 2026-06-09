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


DATA_PATH = "/kaggle/input/playground-series-s5e12"
TARGET = "diagnosed_diabetes"

train = pd.read_csv(f"{DATA_PATH}/train.csv")
test = pd.read_csv(f"{DATA_PATH}/test.csv")
sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

train.head()


train.describe().T


print("train shape: ",train.shape)
print("test shape: ",test.shape)
print("submission shape", sub.shape)


train.info()


train[TARGET].value_counts(normalize=True).rename("ratio")


missing = pd.DataFrame({
    "missing_count" : train.isna().sum(),
    "missing_ratio" : train.isna().mean()
}).sort_values("missing_ratio", ascending=False)

missing.head(15)


id_col = "id"

X = train.drop(columns=[TARGET, id_col])
y = train[TARGET].astype(int)

X_test = test.drop(columns = id_col)

print("X:", X.shape, "y:", y.shape, "X_test:", X_test.shape)
y.value_counts(normalize = True)


cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
print("Categorical columns count:", len(cat_cols))
cat_cols


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

print("Train: ", X_train.shape, "Validation: ", X_valid.shape)
print("y_train mean: ",y_train.mean(), "y_valid mean: ", y_valid.mean())


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

model = CatBoostClassifier(
    iterations = 2000, 
    learning_rate = 0.05,
    depth = 8,
    loss_function = "Logloss",
    eval_metric = "AUC",
    random_seed = 42,
    verbose = 200,
    od_type = "Iter",
    od_wait = 200,
    allow_writing_files = False
)

model.fit(
    X_train, y_train,
    eval_set = (X_valid, y_valid),
    cat_features = cat_cols,
    use_best_model = True
)

valid_pred = model.predict_proba(X_valid)[:, 1]
print("Holdout AUC: ", roc_auc_score(y_valid, valid_pred))
print("Best iteration: ", model.get_best_iteration())


best_iter = model.get_best_iteration()
best_iter = int(best_iter) if best_iter and best_iter > 0 else 1200

final_model = CatBoostClassifier(
    iterations = best_iter,
    learning_rate = 0.05,
    depth = 8,
    loss_function = "Logloss",
    eval_metric = "AUC",
    random_seed = 42,
    verbose = 200,
    allow_writing_files = False
)

final_model.fit(X, y, cat_features=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]

sub[TARGET] = test_pred
sub.to_csv("submission.csv", index = False)

print("Saved: submission.csv")
sub.head()

