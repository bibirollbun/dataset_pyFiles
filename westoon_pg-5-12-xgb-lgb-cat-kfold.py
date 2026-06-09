import numpy as np 
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


train_dir = "/kaggle/input/playground-series-s5e12/train.csv"
test_dir = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(train_dir)
train = train.drop(columns='id')

test = pd.read_csv(test_dir)
test = test.drop(columns='id')


target_col = ['diagnosed_diabetes']
print(f"Target column: {target_col} type: {train[target_col].dtypes.iloc[0]} \n")


X = train.drop(columns=target_col)
y = train[target_col]


num_dt_cols = train.select_dtypes(include=['number']).columns[:-1].to_list() 
cat_dt_cols = train.select_dtypes(include=['object', 'category']).columns.to_list()

potencial_cat_cols = [col for col in train[num_dt_cols] if train[col].nunique() <= 5]
print(f"Potential categorical columns in num_dt_cols: {potencial_cat_cols} \n")

cat_cols = cat_dt_cols + potencial_cat_cols
num_cols = [col for col in num_dt_cols if col not in potencial_cat_cols]

print(f"Numerical columns: {num_cols}\n")
print(f"Categorical columns: {cat_cols}\n")
print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}")  


def train_folds_lgb(X, y, test,
                    model_params=None, fit_params=None,
                    cat_cols=None,
                    n_splits=5, shuffle=True,
                    random_state=777):

    model_params = model_params or {}
    fit_params = fit_params or {}

    cat_cols_idx = [X.columns.get_loc(c) for c in cat_cols] if cat_cols else None
    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype('category')
            test[c] = test[c].astype('category')


    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        model = LGBMClassifier(**model_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols_idx,
            **fit_params
        )


        fold_pred = model.predict_proba(X_val)[:,1]
        oof_preds[val_idx] = fold_pred
        fold_auc = roc_auc_score(y_val, fold_pred)
        print(f"Fold {fold} ROC AUC: {fold_auc:.4f}")

        test_preds += model.predict_proba(test)[:,1]

    test_preds /= n_splits
    mean_auc = roc_auc_score(y, oof_preds)
    print(f"Mean ROC AUC: {mean_auc:.4f}")

    return oof_preds, test_preds


def train_folds_xgb(X, y, test,
                    cat_cols,
                    num_cols,
                    model_params=None,
                    fit_params=None,
                    n_splits=5,
                    shuffle=True,
                    random_state=777):

    model_params = model_params or {}
    fit_params = fit_params or {}

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        preprocess = ColumnTransformer([
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

        X_tr_enc = preprocess.fit_transform(X_tr)
        X_val_enc = preprocess.transform(X_val)
        X_test_enc = preprocess.transform(test)

        model = XGBClassifier(**model_params)

        model.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_val_enc, y_val)],
            **fit_params
        )

        fold_pred = model.predict_proba(X_val_enc)[:, 1]
        oof_preds[val_idx] = fold_pred
        test_preds += model.predict_proba(X_test_enc)[:, 1]

        fold_auc = roc_auc_score(y_val, fold_pred)
        print(f"Fold {fold} ROC AUC: {fold_auc:.4f}")

    test_preds /= kf.get_n_splits()

    mean_auc = roc_auc_score(y, oof_preds)
    print(f"Mean ROC AUC: {mean_auc:.4f}")

    return oof_preds, test_preds


def train_folds_catboost(X, y, test,
                         model_params=None, fit_params=None,
                         cat_cols=None,
                         n_splits=5, shuffle=True, random_state=777):

    model_params = model_params or {}
    fit_params = fit_params or {}

    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype(str)
            test[c] = test[c].astype(str)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        model = CatBoostClassifier(**model_params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            **fit_params
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:,1]
        test_preds += model.predict_proba(test)[:,1]

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold} ROC AUC: {fold_auc:.4f}")

    test_preds /= kf.get_n_splits()
    mean_auc = roc_auc_score(y, oof_preds)
    print(f"Mean ROC AUC: {mean_auc:.4f}")

    return oof_preds, test_preds


lgb_params = dict(
    n_estimators = 1000,
    device = 'gpu',
    objective = 'binary',
    verbosity=-1,
    random_state=777
)

lgb_fit_params = dict(eval_metric = 'auc')

LGB = train_folds_lgb(
            X, y, test,
            model_params=lgb_params,
            fit_params=lgb_fit_params,
            cat_cols=cat_cols)


cat_params = dict(
    iterations=1000,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    metric_period=10,
    task_type='GPU',
    random_state=777
)

fit_params = dict(
    early_stopping_rounds=100
)

CAT = train_folds_catboost(
    X, y, test,
    model_params=cat_params,
    fit_params=fit_params,
    cat_cols=cat_cols
)


xgb_params = dict(
    tree_method='hist',
    device='cuda',
    eval_metric='auc',
    n_estimators=1000,
    random_state=777
)

xgb_fit_params = dict(
    early_stopping_rounds=50,
    verbose=False
)

XGB = train_folds_xgb(X,y,test,
            model_params=xgb_params,
            cat_cols=cat_cols,
            num_cols=num_cols,
            fit_params=xgb_fit_params)


blend_test = (XGB[1] + LGB[1] + CAT[1]) / 3
print(f'Example preds: {blend_test[:5]}')


sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
sub['diagnosed_diabetes'] = blend_test
sub.to_csv('submission.csv', index=False)

