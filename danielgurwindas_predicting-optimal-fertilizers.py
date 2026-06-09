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
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.info()


test.info()


train.isna().sum()


train.isna().sum()


train.head(2)


from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


SEED = 42
np.random.seed(SEED)


def topk_labels_from_proba(classes, proba, k=3):
    """Return a list[str] of space-delimited top-k class names for each row."""
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    topk_labels = [[classes[j] for j in row] for row in topk_idx]
    return [" ".join(labels) for labels in topk_labels]


TARGET = "Fertilizer Name"         # target per competition
ID_COL = "id"


# If sample_submission is present, confirm output column name (usually "Fertilizer Name").
out_col = TARGET
try:
    sample = pd.read_csv("sample_submission.csv")
    non_id_cols = [c for c in sample.columns if c != ID_COL]
    if len(non_id_cols) == 1:
        out_col = non_id_cols[0]
except Exception:
    pass

print(train.head(3))
print(train.columns.tolist())


feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
X = train[feature_cols].copy()
y = train[TARGET].astype(str)   # ensure string labels

X_test = test[feature_cols].copy()

# Identify categorical vs numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
num_cols = [c for c in X.columns if c not in cat_cols]

cat_cols, num_cols



cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

pre = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ],
    remainder="drop"
)

clf = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=None,
    max_iter=400,
    l2_regularization=0.0,
    random_state=SEED
)

pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])



pipe.fit(X, y)
classes = pipe.named_steps["clf"].classes_.tolist()
classes[:5], len(classes)  # quick peek



proba_test = pipe.predict_proba(X_test)
pred_top3 = topk_labels_from_proba(classes, proba_test, k=3)



submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    out_col: pred_top3
})
submission.to_csv("submission.csv", index=False)


import os
print("Saved at:", os.path.abspath("submission.csv"))





