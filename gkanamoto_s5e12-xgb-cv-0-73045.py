import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, time

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


#--- Configurations
SEED = 42
FOLDS = 10

def seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# --- Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print("train", train_df.shape, "test", test_df.shape)


cols = [col for col in train_df.columns if col not in ['id', 'diagnosed_diabetes']]
new_cols = []

for col in cols:
    # mean
    mean_map = orig.groupby(col)['diagnosed_diabetes'].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name

    train_df = train_df.merge(mean_map, on=col, how='left')
    test_df = test_df.merge(mean_map, on=col, how='left')
    new_cols.append(new_mean_col_name)

    # count
    new_cnt_col_name = f"orig_cnt_{col}"
    cnt_map = orig.groupby(col).size().reset_index(name=new_cnt_col_name)

    train_df = train_df.merge(cnt_map, on=col, how='left')
    test_df = test_df.merge(cnt_map, on=col, how='left')
    new_cols.append(new_cnt_col_name)

for col in new_cols:
    if 'mean' in col:
        train_df[col] = train_df[col].fillna(orig['diagnosed_diabetes'].mean())
        test_df[col] = test_df[col].fillna(orig['diagnosed_diabetes'].mean())
    else:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)


#--- handling categorical features
from sklearn.preprocessing import OneHotEncoder

cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# train data
encoded_train = encoder.fit_transform(train_df[cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cols))
train_df = pd.concat([train_df.drop(columns=cols), encoded_train_df], axis=1)

# test data
encoded_test = encoder.transform(test_df[cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cols))
test_df = pd.concat([test_df.drop(columns=cols), encoded_test_df], axis=1)    


X = train_df.drop(columns=['id', 'diagnosed_diabetes'])
y = train_df['diagnosed_diabetes']


import optuna
from sklearn.model_selection import StratifiedKFold

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",       # GPU
        "predictor": "gpu_predictor",
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Stratified KFold
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    auc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val   = X.iloc[val_idx]
        y_val   = y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        preds = model.predict(dval)
        auc_scores.append(roc_auc_score(y_val, preds))

    return sum(auc_scores) / len(auc_scores)


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)
# best_params = study.best_params

# print("Best AUC:", study.best_value)
# print("Best Params:", best_params)


best_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",       # GPU
        "predictor": "gpu_predictor",
        "lambda": 0.025054758350544867,
        "alpha": 0.15945687800025848,
        "learning_rate": 0.06672550934250764,
        "max_depth": 3,
        "subsample": 0.7314797824448713,
        "colsample_bytree": 0.5694660231098494,
        "min_child_weight":8,
    }


roc_scores = []
models = []
# params = {
#     "objective": "binary:logistic",
#     "eval_metric": "auc", 
#     "learning_rate": 0.05,
#     "max_depth": 6,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "seed": 42,
# }

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"========= \nFold {fold+1}/{FOLDS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = best_params
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0
    )

    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))
    
    # Calculate the score
    score = roc_auc_score(y_val, y_pred_val)
    print(f'Fold: {fold+1} AUC score: {np.mean(score):.5f}') 

    roc_scores.append(score)
    models.append(model)


print(f'\nAverage AUC Score : {np.mean(roc_scores):.5f}, +-: {np.std(roc_scores):.5f}')


test_id = test_df.id
X_test = test_df.drop(columns=["id"])
submit_score = []

dtest = xgb.DMatrix(X_test)
for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest, iteration_range=(0, model.best_iteration))
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)


submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': pred
})

# Save
submission.to_csv('submission.csv', index=False)


submission

