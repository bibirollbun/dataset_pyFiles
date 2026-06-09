import os
import time 
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import catboost as cb
from tqdm import tqdm


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class CFG:
    debug = False
    random_seed = 42
    n_splits = 5
    target = "Exited"
    drop_col = ["id"]#, "CustomerId"]
    data_dir = "../input/"
    exp_name = "catboost_baseline"
    cat_col = ["Gender", "Geography", "HasCrCard", "IsActiveMember", "Surname"]


class catboost_CFG:
    iterations = 10000  # 最大イテレーション数
    early_stopping_rounds = 10  # 早期終了の設定
    learning_rate = 0.01
    depth = 6  # 決定木の深さ
    cat_features = [1, 3, 4, 9, 10]



train_df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "test.csv"))
sub = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "sample_submission.csv"))


"""vocab = {}
for name in train_df["Surname"].values.tolist():
    name = name.lower()
    for i in range(len(name)):
        for j in range(len(name)-i):
            word = name[i:i+j]
            if word in vocab.keys():
                vocab[word] += 1
            else:
                vocab[word] = 0
new_vocab = {}
vocab_key = vocab.keys()

for word in tqdm(vocab_key):
    is_unique = True
    for key in vocab_key:
        if word in key and vocab[key]==vocab[word] and word!=key and len(key) > len(word):
            is_unique = False
            break

    if is_unique and vocab[word] > 100 and (len(word) > 3 or len(word)==1):
        new_vocab[word] = vocab[word]"""


def process_data(df, is_train=True):
    """vocab = set(new_vocab)
    temp_df = pd.DataFrame(0, index=df.index, columns=[f"num_{word}" for word in vocab])

    for id, name in tqdm(enumerate(df["Surname"].values.tolist())):
        name = name.lower()
        for token in vocab:
            if token in name:
                temp_df.at[id, f"num_{token}"] += 1

    df = pd.concat([df, temp_df], axis=1)"""

    for number in [2, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
        df[f"mod_{number}"] = (df["CustomerId"] % number).astype(int)

    for cat_col in CFG.cat_col:
        df[cat_col] = df[cat_col].astype(str)

    df["start_age"] = df["Age"] - df["Tenure"]
    df["salary_ratio"] = df["Balance"] / df["EstimatedSalary"]

    df = df.drop(CFG.drop_col, axis=1)

    if is_train:
        return df.drop(CFG.target, axis=1), df[CFG.target]
    else:
        return df


#print(len(set(new_vocab)))


train_df, train_target = process_data(train_df)
test_df = process_data(test_df, is_train=False)


def train_catboost_one_fold(X_train, y_train, X_valid, y_valid, X_test, params, fold_id):
    print(f"====================training fold {fold_id}====================")
    
    model = cb.CatBoostClassifier(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        cat_features=params['cat_features'],
        early_stopping_rounds=params['early_stopping_rounds'],
        verbose=100
    )
    
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=100)
    
    oof = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, oof)
    print(f"fold {fold_id} roc-auc: {score}")
    
    test_preds = model.predict_proba(X_test)[:, 1]
    return oof, test_preds, model

def train_catboost(X, y, test, params):
    oof_preds = np.zeros((X.shape[0], ))
    test_preds = np.zeros((test.shape[0], ))
    
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    feature_importance_df = pd.DataFrame()
    
    for fold_id, (train_index, valid_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        oof, test_pred, model = train_catboost_one_fold(X_train, y_train, X_valid, y_valid, test, params, fold_id)
        
        oof_preds[valid_index] = oof
        test_preds += test_pred / CFG.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = X.columns
        fold_importance_df["Importance"] = model.get_feature_importance()
        fold_importance_df["Fold"] = fold_id
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    score = roc_auc_score(y, oof_preds)
    print(f"Total ROC-AUC: {score:.5f}")
    
    plot_feature_importance(feature_importance_df)
    
    return oof_preds, test_preds

def plot_feature_importance(feature_importance_df):
    importance_mean = feature_importance_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    importance_mean[:20].plot(kind='barh', color='blue', edgecolor='k')
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importance")
    plt.show()

params = {
    'iterations': catboost_CFG.iterations,
    'learning_rate': catboost_CFG.learning_rate,
    'depth': catboost_CFG.depth,
    'cat_features': catboost_CFG.cat_features,
    'early_stopping_rounds': catboost_CFG.early_stopping_rounds
}


train_df


test_df


def main():
    oof, test_preds = train_catboost(train_df, train_target, test_df, params)
    sub[CFG.target] = test_preds
    sub.to_csv(f"submission_{CFG.exp_name}.csv", index=False)
    df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "train.csv"))
    oof_df = pd.DataFrame({"id":df["id"], "oof":oof.tolist()})
    oof_df.to_csv(f"oof_{CFG.exp_name}.csv", index=False)

if __name__ == "__main__":
    main()




