!pip install scikit-learn==1.4.2


import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as catb
from catboost import CatBoostClassifier

from sklearn.base import clone

import optuna

from typing import Dict

import warnings
warnings.filterwarnings("ignore")


RANDOM_STATE = 42


def cross_val_skf(model,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  n_splits: int=5,
                  random_state: int=None) -> float:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    
    for i, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]        
        y_pred = []
        
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      verbose=0)
            
            best_iter = model.best_iteration + 1
            y_pred = model.predict(X_vl, iteration_range=(0, best_iter))
        elif isinstance(model, lgb.LGBMClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=50,
                                             verbose=0)
                      ])
            
            best_iter = model.best_iteration_ + 1
            y_pred = model.predict(X_vl , num_iteration=best_iter)
        elif isinstance(model, catb.CatBoostClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      use_best_model=True,
                      verbose=0)
            
            y_pred = model.predict(X_vl)
        else:
            model.fit(X_tr,
                      y_tr)
            
            y_pred = model.predict(X_vl)
        
        acc = accuracy_score(y_vl, y_pred)
        scores.append(acc)
        
        print(f"FOLD {i} | Accuracy: {acc:.5f}")
    
    avg_acc = round(np.mean(scores), 5)
    print(f"\nAvarege Accurary: {avg_acc:.5f}\n")
    
    return avg_acc

def oof_prediction(model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   n_splits: int=5,
                   random_state: int=None) -> Dict[str, np.ndarray]:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_train = np.zeros(len(X_train))
    oof_test = np.zeros(len(X_test))
    
    for tr_idx, vl_idx in skf.split(X_train, y_train):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]        
        
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      verbose=0)
            
            best_iter = model.best_iteration + 1
            oof_train[vl_idx] = model.predict_proba(X_vl, iteration_range=(0, best_iter))[:, 1]
            oof_test += model.predict_proba(X_test, iteration_range=(0, best_iter))[:, 1] / n_splits
        elif isinstance(model, lgb.LGBMClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=50,
                                             verbose=0)
                      ])
            
            best_iter = model.best_iteration_ + 1
            oof_train[vl_idx] = model.predict_proba(X_vl , num_iteration=best_iter)[:, 1]
            oof_test += model.predict_proba(X_test , num_iteration=best_iter)[:, 1] / n_splits
        elif isinstance(model, catb.CatBoostClassifier):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      use_best_model=True,
                      verbose=0)
            
            oof_train[vl_idx] = model.predict_proba(X_vl)[:, 1]
            oof_test += model.predict_proba(X_test)[:, 1] / n_splits
        else:
            model.fit(X_tr,
                      y_tr)
            
            oof_train[vl_idx] = model.predict_proba(X_vl)[:, 1]
            oof_test += model.predict_proba(X_test)[:, 1] / n_splits  
        
    return {"train": oof_train,
            "test": oof_test}


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


print("#" * 40 + " Train dataset " + "#" * 40)
display(train.head())
train.info()
display(train.describe())
display(train.isna().sum())
print("\n" + "#" * 40 + " Test dataset " + "#" * 40)
display(test.head())
test.info()
display(test.describe())
display(test.isna().sum())


X = train.drop("Personality", axis=1)
y = train["Personality"]
X_test = test.copy()

num_features = X.select_dtypes(["int64", "float64"]).columns.to_list()
cat_features = X.select_dtypes(["object"]).columns.to_list()


preprocesor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant",
                                  fill_value="missing")),
        ("encoder", OrdinalEncoder())
    ]), cat_features)
])

target_preprocesor = LabelEncoder()

X_train = preprocesor.fit_transform(X)
X_test = preprocesor.transform(X_test)

X_train = pd.DataFrame(X_train, columns=preprocesor.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=preprocesor.get_feature_names_out())

cat_idx = [i for (i, col) in enumerate(preprocesor.get_feature_names_out()) if col.split("_")[0] == "cat"]
cat_col = [X_train.columns[idx] for idx in cat_idx]

for col in cat_col:
    X_train[col] = X_train[col].astype("int64")
    X_test[col] = X_test[col].astype("int64")
    
y_train = target_preprocesor.fit_transform(y)


# hyperparameters (selected Optuna)
catb_params0 = {"loss_function": "Logloss",
                "eval_metric": "Logloss",
                'iterations': 317,
                'depth': 6,
                'learning_rate': 0.012506922242313949,
                'reg_lambda': 0.04507781593406735,
                'min_child_samples': 7,
                'scale_pos_weight': 1.3542115856321328,
                "task_type": "GPU",
                "early_stopping_rounds": 50,
                "random_state": 42,}

lgb_params0 = {"objective": "binary",
               "metric": "binary_logloss",
               'n_estimators': 634,
               'max_depth': 9,
               'learning_rate': 0.04393594470568061,
               'min_child_weight': 6,
               'min_split_gain ': 0.0035453310811935433,
               'sub_sample': 0.9059823849047283,
               'colsample_bytree': 0.5429279484488345,
               'reg_alpha': 0.01726798858257095,
               'reg_lambda': 0.031286515687581715,
               'scale_pos_weight': 1.5834025562711627,
               "verbose": -1,
               "device_type": "gpu",
               "random_state": 42}

xgb_params0 = {"objective": "binary:logistic",
               "eval_metric": "logloss",
               'n_estimators': 382,
               'max_depth': 4,
               'max_bin': 671,
               'learning_rate': 0.0011878176963551427,
               'sub_sample': 0.9438273672535726,
               'min_child_weight': 7,
               'gamma': 0.0019107144882749613,
               'colsample_bytree': 0.5576075825184463,
               'reg_alpha': 0.022918483134493343,
               'reg_lambda': 0.0010021862343176035,
               'scale_pos_weight': 2.31447347695404,
               "tree_method": "hist",
               "device": "cuda",
               "early_stopping_rounds": 100,
               "random_state": 4}


hgb_params0 = {'learning_rate': 0.0020590007192032712,
               'learning_rate': 0.0016973485406240752,
               'max_iter': 961,
               'max_leaf_nodes': 197,
               'max_depth': 4,
               'min_samples_leaf': 58,
               'l2_regularization': 0.010865616090637922,
               'max_features': 0.0017557546717319919,
               'max_bins': 35,
               'class_weight': None,
               "early_stopping": False,
               "random_state": 42}


models = [{"name": "xgb", "estimator": XGBClassifier(**xgb_params0)},
          {"name": "lgb", "estimator": LGBMClassifier(**lgb_params0)},
          {"name": "catb", "estimator": CatBoostClassifier(**catb_params0)},
          {"name": "hgb", "estimator": HistGradientBoostingClassifier(**hgb_params0)}]
scores_models = {}

for model in models:
    name = model["name"]
    estimator = clone(model["estimator"])
    
    print(f"Model {name}")
    scores_models[name] = cross_val_skf(estimator,
                                        X_train.values,
                                        y_train,
                                        random_state=RANDOM_STATE)
    
print(scores_models)


oof_preds_models = {}

for model in models:
    name = model["name"]
    estimator = clone(model["estimator"])
    
    oof_preds_models[name] = oof_prediction(estimator,
                                            X_train.values,
                                            y_train,
                                            X_test.values,
                                            random_state=RANDOM_STATE)


oof_train = np.array([oof_preds["train"] for oof_preds in oof_preds_models.values()]).T
oof_test = np.array([oof_preds["test"] for oof_preds in oof_preds_models.values()]).T


# Ensemble weights and threshold (selected by Optuna)
weights = [-0.5578451889320368, 0.6283346974333689, 0.6927092210002257, 0.9803073100277904]
threshold = 0.5802099265890668
s_weights = weights / np.sum(weights)
w_probs = np.average(oof_test, axis=1, weights=s_weights)
y_pred = (w_probs > threshold).astype(np.int64)


submission["Personality"] = np.where(y_pred == 0, "Extrovert", "Introvert")
submission.to_csv("weighted_ensembling.csv", index=False)
submission.head()

