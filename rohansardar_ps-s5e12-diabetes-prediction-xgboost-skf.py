import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train.head()


train.info()


test.info()


target = "diagnosed_diabetes"
idx = "id"

X = train.drop(columns=[target, idx])
y = train[target]
test = test.drop(columns=[idx])


cat_cols = [c for c in X.columns if X[c].dtype == "object"]

print("Categorical columns:", cat_cols)

for c in cat_cols:
    combined = pd.concat([X[c], test[c]], axis=0)
    codes, uniques = pd.factorize(combined)

    X[c] = codes[:len(X)]
    test[c] = codes[len(X):]


xgb_params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05488248216649515,
    'max_depth': 3,
    'min_child_weight': 9,
    'subsample': 0.9324156371717225,
    'gamma': 0.5502813883351498,
    'reg_alpha': 2.755473920545449e-07,
    'colsample_bytree': 0.5974622135152583,
    'reg_lambda': 3.1929676333640495,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'
}


params = xgb_params


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===================== FOLD {fold} =====================")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=200,
        verbose_eval=400,
    )

    oof[val_idx] = model.predict(
        dvalid,
        iteration_range=(0, model.best_iteration + 1)
    )

    test_preds += model.predict(
        dtest,
        iteration_range=(0, model.best_iteration + 1)
    ) / skf.n_splits


print("\nOOF ROC-AUC:", roc_auc_score(y, oof))

sample_sub[target] = test_preds
sample_sub.to_csv("submission.csv", index=False)




