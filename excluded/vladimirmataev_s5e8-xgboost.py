import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from category_encoders import TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

import xgboost as xgb
from xgboost import XGBClassifier

import cupy as cp


SEED = 44
N_SPLITS = 10

train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path = "/kaggle/input/playground-series-s5e8/test.csv"
sub_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"
original_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"


def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    
    df["arcsinh_balance"] = np.arcsinh(df["balance"])
    df["arcsinh_duration"] = np.arcsinh(df["duration"])
    df["balance/age"] = df["balance"] * df["age"]
    df["arcsinh_balance/age"] = df["arcsinh_balance"] * df["age"]
    
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df["month_num"] = df["month"].map(month_map).astype("int64")
    df = df.drop("month", axis=1)
    df = df.rename(columns={"month_num": "month"})
    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df

def cross_val_skf_and_pred(model,
                           X_train: cp.array,
                           y_train: cp.array,
                           X_test: cp.array,                  
                           n_splits: int=None,
                           shuffle: bool=True):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED)
    oof_train = np.zeros(len(X_train))
    oof_test = np.zeros(len(X_test))
    scores = []

    X_train_cpu = cp.asnumpy(X_train)
    y_train_cpu = cp.asnumpy(y_train)
    
    for i, (tr_idx, vl_idx) in enumerate(skf.split(X_train_cpu, y_train_cpu), 1):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
        
        model.fit(X_tr,
                y_tr,
                eval_set=[(X_vl, y_vl)],
                verbose=False)
        
        best_iter = model.best_iteration + 1
        y_probs = model.predict_proba(X_vl, iteration_range=(0, best_iter))[:, 1]
        oof_train[vl_idx] = y_probs
        oof_test += model.predict_proba(X_test, iteration_range=(0, best_iter))[:, 1] / n_splits
        
        auc = roc_auc_score(y_vl.get(), y_probs)
        scores.append(auc)
        
        print(f"Fold {i}/{n_splits} | Roc Auc: {auc}")
        
    return {"scores": scores,
            "train": oof_train,
            "test": oof_test}


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submmission = pd.read_csv(sub_path)
original = pd.read_csv(original_path, sep=";")

print("\n-----Train data-----")
display(train.head())
train.info()
print(train.isna().sum())
display(train.describe())
print("\n-----Original data-----")
display(original.head())
original.info()
print(original.isna().sum())
display(original.describe())
print("\n-----Test data-----")
display(test.head())
test.info()
print(test.isna().sum())
display(test.describe())



original["y"] = original["y"].map({"no": 0, "yes": 1})
all_data = pd.concat([train, original, test], ignore_index=True)

all_data = add_features(all_data)

all_y = all_data["y"]
all_data = all_data.drop(["id", "y"], axis=1)

num_features = all_data.select_dtypes(include=np.number).columns.to_list()
cat_features = all_data.select_dtypes("object").columns.to_list()

train_and_orig = all_data.iloc[:len(train)+len(original)]
y = all_y[:len(train)+len(original)]
X_test = all_data.iloc[-len(test):]

preprocer = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", TargetEncoder(), cat_features)
])

X_train = preprocer.fit_transform(train_and_orig, y)
y_train = np.array(y)
X_test = preprocer.transform(X_test)

X_train.shape, y_train.shape, X_test.shape


X_train_gpu = cp.array(X_train, dtype=cp.float32)
y_train_gpu = cp.array(y_train, dtype=cp.float32)
X_test_gpu = cp.array(X_test, dtype=cp.float32)


# hyperparameters (selected Optuna)
xgb_params = {"objective": "binary:logistic",
               "eval_metric": "auc",
               "n_estimators": 1421,
               "max_bin": 11619,
               "max_depth": 9,
               "learning_rate": 0.060542156678920725,
               "subsample": 0.8860705042275745,
               "colsample_bytree": 0.5890358175215191,
               "reg_alpha": 0.004559752117634602,
               "reg_lambda": 0.004194941299345613,
               "min_child_weight": 5,
               "gamma": 0.2226553985484643,
               "scale_pos_weight": 1.2631169451535507,
               "grow_policy" : "lossguide",
               "tree_method": "hist",
               "early_stopping_rounds": 100,
               "device": "cuda",
               "seed": SEED}


model = XGBClassifier(**xgb_params)

results = cross_val_skf_and_pred(model,
                                 X_train_gpu,
                                 y_train_gpu,
                                 X_test_gpu,
                                 N_SPLITS)

print(f"Average roc auc: {np.mean(results['scores'])}")


feature_importance = pd.DataFrame({"feature": preprocer.get_feature_names_out(),
                                   "importance": model.feature_importances_})
feature_importance = feature_importance.sort_values(by="importance", ascending=False).iloc[:20]

sns.barplot(x=feature_importance["importance"], y=feature_importance["feature"])


submmission["y"] = results["test"]
submmission.to_csv("xgb.csv", index=False)
submmission.head()

