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


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

# LightGBM
from lightgbm import LGBMClassifier

# CatBoost
from catboost import CatBoostClassifier, Pool


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")

target = 'loan_paid_back'
drop_features = ['id']
X = train.drop(columns=[target] + drop_features, errors='ignore')
y = train[target].astype(int)  # pastikan biner 0/1


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))



from itertools import combinations

def add_features(X):
    # kumpulkan kolom baru di dict
    new_cols = {}

    #changer numeric to categorical
    X['credit_band'] = pd.cut(X['credit_score'],
                               bins=[0, 600, 660, 720, 780, 1000],
                               labels=['subprime', 'near-prime', 'prime', 'super-prime', 'elite'])

    X['disposable_income_proxy'] = X['annual_income'] * (1 - X['debt_to_income_ratio'])
    
    # numeric features
    base_numeric_features = X.select_dtypes(include=[np.number]).columns
    for col1, col2 in combinations(base_numeric_features, 2):
        new_cols[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        new_cols[f'{col1}_d_{col2}'] = X[col1] / (X[col2] + 1e-10)

    # categorical features
    base_categorical_features = X.select_dtypes(include=['object', 'category']).columns
    for col1, col2 in (combinations(base_categorical_features, 2)):
        new_cols[f'{col1}__{col2}'] = X[col1].astype(str) + "_" + X[col2].astype(str)

    # gabungkan sekaligus → hindari fragmentasi
    X_new = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

    return X_new

X_train_fe = add_features(X_train)
X_valid_fe = add_features(X_valid)

# Update categorical list with engineered categorical feature
num_features_fe = X_train_fe.select_dtypes(include=[np.number]).columns
cat_features_fe = X_train_fe.select_dtypes(include=['object', 'category']).columns


# -----------------------------
# Model — CatBoost (native categorical)
# -----------------------------
X_train_cb = X_train_fe.copy()
X_valid_cb = X_valid_fe.copy()

# Ensure categorical dtypes
for c in cat_features_fe:
    X_train_cb[c] = X_train_cb[c].astype('category')
    X_valid_cb[c] = X_valid_cb[c].astype('category')

cat_indices = [X_train_cb.columns.get_loc(c) for c in cat_features_fe]

cb = CatBoostClassifier(
    iterations=5000,            # large with early stopping
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=8.0,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    od_type='Iter',             # overfitting detector (early stopping)
    od_wait=50,
    subsample=0.8,
    rsm=0.8,                    # column sampling
    thread_count=-1,
    class_weights=[class_weight_dict.get(0, 1.0), class_weight_dict.get(1, 1.0)]
)

# Training Process
train_pool = Pool(X_train_cb, y_train, cat_features=cat_indices)
valid_pool = Pool(X_valid_cb, y_valid, cat_features=cat_indices)

cb.fit(train_pool, eval_set=valid_pool, verbose=1000)


cb_valid_pred = cb.predict_proba(valid_pool)[:, 1]
cb_auc = roc_auc_score(y_valid, cb_valid_pred)
print(f"[CatBoost] Valid ROC AUC: {cb_auc:.5f}")


subdf = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
subdf_id=subdf['id']
X_subdf=subdf.drop(columns=['id'], errors='ignore')
X_subdf_fe = add_features(X_subdf)

X_subdf_cb = X_subdf_fe.copy()

for c in cat_features_fe:
    X_subdf_cb[c] = X_subdf_cb[c].astype('category')

subdf_pool = Pool(X_subdf_cb,cat_features=cat_indices)
cb_subdf_pred = cb.predict(subdf_pool)



submission = pd.DataFrame({
    "id": subdf_id,
    "loan_paid_back": cb_subdf_pred
})

submission_prob.to_csv("/kaggle/working/submission_prob.csv", index=False)

