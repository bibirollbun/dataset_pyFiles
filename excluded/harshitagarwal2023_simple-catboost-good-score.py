import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


target = 'diagnosed_diabetes'
test_id = test['id'].copy()


X = train.drop(target, axis=1)
y = train[target]
test_X = test.copy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)


cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.4,
    "min_data_in_leaf": 30,
    "od_type": "Iter",
    "od_wait": 300,
    "random_seed": 42,
    "verbose": 0,
    "task_type": "GPU",
    "devices": "0,1",
    "boosting_type": "Ordered",
    "random_strength": 2.0,
    "border_count": 254
}


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_X))
fold_auc = []


cat_cols = X.select_dtypes(include=['object', 'category']).columns
cat_features = [X.columns.get_loc(col) for col in cat_cols]


N_SPLITS = 7
skf = StratifiedKFold(n_splits = N_SPLITS, shuffle=True, random_state = 42)


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )

    val_pool = Pool(
        X_val,
        y_val,
        cat_features=cat_features
    )

    model = CatBoostClassifier(**cat_params)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=200
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred

    auc = roc_auc_score(y_val, val_pred)
    fold_auc.append(auc)
    print(f"AUC: {auc:.5f}")

    test_preds += model.predict_proba(test_X)[:, 1] / N_SPLITS



pd.DataFrame({
    'id' : test_id,
    target : test_preds
}).to_csv('Single_CB_submission.csv', index=False)




