import os, gc, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
target = "diagnosed_diabetes"
id_c = "id"
n_folds = 5
random_seed = 42
verbose = 100


def seed_everything(seed=random_seed):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
seed_everything()


train_ids = train[id_c].copy()
test_ids = test[id_c].copy()


y =  train[target].values
train.drop([target], axis=1, inplace=True)


threshold = 0.5
missing_frac = train.isna().mean()
drop_cols = missing_frac[missing_frac > threshold].index.tolist()
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)


common_cols = [ c for c in train.columns if c in test.columns and c != id_c]
X = train[common_cols].copy()
X_test = test[common_cols].copy()


num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


num_imputer = SimpleImputer(strategy='median')
X_num = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)


freq_encoded = []
onehot_cols = []
for c in cat_cols:
    nunique = X[c].nunique()
    if nunique <= 10:
        onehot_cols.append(c)
    else:
        freq = X[c].value_counts(dropna=False) / len(X)
        X_num[c + "_freq"] = X[c].map(freq).fillna(0).values
        X_test_num[c + "_freq"] = X_test[c].map(freq).fillna(0).values
        freq_encoded.append(c)


if onehot_cols:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe.fit(pd.concat([X[onehot_cols], X_test[onehot_cols]], axis=0))
    ohe_cols_train = pd.DataFrame(ohe.transform(X[onehot_cols]), index=X.index)
    ohe_cols_test  = pd.DataFrame(ohe.transform(X_test[onehot_cols]), index=X_test.index)
    # name columns
    ohe_cols_train.columns = [f"ohe_{i}" for i in range(ohe_cols_train.shape[1])]
    ohe_cols_test.columns = ohe_cols_train.columns
    X_proc = pd.concat([X_num, ohe_cols_train], axis=1)
    X_test_proc = pd.concat([X_test_num, ohe_cols_test], axis=1)
else:
    X_proc = X_num.copy()
    X_test_proc = X_test_num.copy()


features = X_proc.columns.tolist()


# ---------- CV setup ----------
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
oof_preds = np.zeros(len(X_proc))
test_preds_lgb = np.zeros(len(X_test_proc))
test_preds_xgb = np.zeros(len(X_test_proc))
test_preds_cat = np.zeros(len(X_test_proc))


# ---------- Train LightGBM, XGBoost, CatBoost ----------
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_proc, y)):
    print(f"Fold {fold+1}/{n_folds}")
    X_tr, X_val = X_proc.iloc[tr_idx], X_proc.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': random_seed + fold,
        'verbosity': -1,
        'early_stopping_rounds': 100
    }
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val]
    )
    val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_preds_lgb += lgb_model.predict(X_test_proc, num_iteration=lgb_model.best_iteration) / n_folds

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, use_label_encoder=False, eval_metric='auc',
        random_state=random_seed + fold
    )
    xgb_clf.fit(X_tr, y_tr, early_stopping_rounds=100, eval_set=[(X_val, y_val)], verbose=verbose)
    test_preds_xgb += xgb_clf.predict_proba(X_test_proc)[:,1] / n_folds
    val_pred_xgb = xgb_clf.predict_proba(X_val)[:,1]

    # CatBoost (handles categorical but we already freq/ohe encoded; still good)
    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, eval_metric='AUC',
        random_seed=random_seed + fold, verbose=verbose
    )
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    test_preds_cat += cat.predict_proba(X_test_proc)[:,1] / n_folds
    val_pred_cat = cat.predict_proba(X_val)[:,1]

    # Simple average of GBM predictions for this fold (stacking features)
    oof_preds[val_idx] = (val_pred_lgb + val_pred_xgb + val_pred_cat) / 3

    # cleanup
    del lgb_model, xgb_clf, cat, val_pred_lgb, val_pred_xgb, val_pred_cat
    gc.collect()

print("OOF ROC AUC:", roc_auc_score(y, oof_preds))


# ---------- Stacking (meta-model) ----------
# Create train-level stacking features by retraining base models with OOF approach above.
# Here we already have oof_preds as single blended feature. We can form a simple meta matrix:
meta_X = np.vstack([oof_preds]).T
# For test, combine the averaged test preds:
meta_test = np.vstack([(test_preds_lgb + test_preds_xgb + test_preds_cat) / 3 ]).T


# Fit a simple logistic regression meta model
meta_clf = LogisticRegression()
meta_clf.fit(meta_X, y)
final_test_preds = meta_clf.predict_proba(meta_test)[:,1]


# ---------- Prepare submission ----------
submission = pd.DataFrame({id_c: test_ids, target: final_test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

