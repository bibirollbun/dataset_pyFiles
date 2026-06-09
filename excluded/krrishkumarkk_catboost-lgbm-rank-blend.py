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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

TARGET = "diagnosed_diabetes"
test_ids = test_df["id"]

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
X_test = test_df.copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 6,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.8,
    "min_data_in_leaf": 50,
    "random_seed": 42,
    "verbose": 200,
    "task_type": "GPU"   # TURN GPU ON
}

for fold, (tr, val) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")
    tr_pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_cols)
    val_pool = Pool(X.iloc[val], y.iloc[val], cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(tr_pool, eval_set=val_pool, use_best_model=True)

    oof[val] = model.predict_proba(X.iloc[val])[:,1]
    test_preds += model.predict_proba(test_pool)[:,1] / 5

print("OOF AUC:", roc_auc_score(y, oof))

submission = pd.DataFrame({
    "id": test_ids,
    TARGET: test_preds
})
submission.to_csv("catboost_native.csv", index=False)



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import CatBoostEncoder

train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

TARGET = "diagnosed_diabetes"
test_ids = test_df["id"]

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
X_test = test_df.copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

encoder = CatBoostEncoder(cols=cat_cols, random_state=42)
X_enc = encoder.fit_transform(X, y)
X_test_enc = encoder.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(skf.split(X_enc, y)):
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42
    )

    model.fit(X_enc.iloc[tr], y.iloc[tr])
    oof[val] = model.predict_proba(X_enc.iloc[val])[:,1]
    test_preds += model.predict_proba(X_test_enc)[:,1] / 5

print("OOF AUC:", roc_auc_score(y, oof))

pd.DataFrame({
    "id": test_ids,
    TARGET: test_preds
}).to_csv("lgb_catboost_encoder.csv", index=False)



import pandas as pd
from scipy.stats import rankdata

cat = pd.read_csv("catboost_native.csv")
lgb = pd.read_csv("lgb_catboost_encoder.csv")

cat["rank"] = rankdata(cat["diagnosed_diabetes"])
lgb["rank"] = rankdata(lgb["diagnosed_diabetes"])

cat["diagnosed_diabetes"] = 0.5 * cat["rank"] + 0.5 * lgb["rank"]
cat["diagnosed_diabetes"] /= cat["diagnosed_diabetes"].max()

cat[["id", "diagnosed_diabetes"]].to_csv("rank_blend.csv", index=False)





