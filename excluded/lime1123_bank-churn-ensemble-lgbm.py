import lightgbm as lgb
import pandas as pd
import glob
import numpy as np
import os
import time 
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb

from tqdm import tqdm
from pprint import pprint


class CFG:
    n_splits = 5
    random_seed = 2025
    data_dir = "../input/"
    target = "Exited"
    oof_path = glob.glob(data_dir + "bank-churn-emsemble/*/oof*.csv")
    submission_path = glob.glob(data_dir + "bank-churn-emsemble/*/submission*.csv")
    oof_path.sort()
    submission_path.sort()
    exp_name = "lgbm-emsemble"
    drop_col = ["id", "CustomerId", "Surname"]
    cat_col = ["Gender"]#, "Surname"]
    onehot_col = ["Geography"]

class lgbm_CFG : 
    # ここらへんは変えない方がいいかも？
    print_freq = 100 # 何ラウンドごとに評価するか
    num_rounds = 10000 # 何ラウンド回すか、大きめの値を入れておいて早期終了する
    early_stopping_rounds = 100 # 何ラウンド改善がなかったら終了するか

cfg = CFG()


pprint(cfg.oof_path)


train_df = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
oof_df = pd.concat([pd.read_csv(oof_path)["oof"] for oof_path in cfg.oof_path], axis=1)
oof_df.columns = [f"oof{i}" for i in range(len(cfg.oof_path))]

oof_df["id"] = train_df["id"]
train_df = train_df.merge(oof_df)

test_df = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
oof_df = pd.concat([pd.read_csv(submission_path)["Exited"] for submission_path in cfg.submission_path], axis=1)
oof_df.columns = [f"oof{i}" for i in range(len(cfg.oof_path))]
oof_df["id"] = test_df["id"]
test_df = test_df.merge(oof_df)

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

    if is_unique and vocab[word] > 100 and (len(word) > 1 or len(word)==1):
        new_vocab[word] = vocab[word]"""


#len(set(new_vocab))


def process_data(df, is_train=True):
    """vocab = set(new_vocab)
    temp_df = pd.DataFrame(0, index=df.index, columns=[f"num_{word}" for word in vocab])

    for id, name in tqdm(enumerate(df["Surname"].values.tolist())):
        name = name.lower()
        for token in vocab:
            if token in name:
                temp_df.at[id, f"num_{token}"] += 1

    df = pd.concat([df, temp_df], axis=1)"""

    for col in CFG.cat_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=CFG.onehot_col, dtype=int)
    
    for number in [2]:
        df[f"mod_{number}"] = (df["CustomerId"] % number).astype(int)
    
    df["start_age"] = df["Age"] - df["Tenure"]
    df["salary_ratio"] = df["Balance"] / df["EstimatedSalary"]
    
    df = df.drop(CFG.drop_col, axis=1)
    
    if is_train:
        return df.drop(CFG.target, axis=1), df[CFG.target]
    else:
        return df



train_df, train_target = process_data(train_df)
test_df = process_data(test_df, is_train=False)


def train_lgbm_one_fold(X_train, y_train, X_valid, y_valid, X_test, params, fold_id) :
    print(f"====================training fold {fold_id}====================")
    # lightgbm 用のデータセットに変換する
    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_valid, label=y_valid)
    
    # モデルの学習、valid を監視して early stopping する
    model = lgb.train(
        params=params,
        train_set=train,
        num_boost_round=lgbm_CFG.num_rounds,
        valid_sets=[train, valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(lgbm_CFG.early_stopping_rounds, verbose=True), lgb.log_evaluation(period=lgbm_CFG.print_freq)],
    )
    
    # oof の結果を返す
    oof = model.predict(X_valid)
    score = roc_auc_score(y_valid, oof)
    print(f"fold {fold_id} roc-auc: {score}")
    
    test_preds = model.predict(X_test)
    return oof, test_preds, model

def train_lgbm(X, y, test, params):
    # oof の結果も返す
    oof_preds = np.zeros((X.shape[0], ))
    test_preds = np.zeros((test.shape[0], ))
    
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    feature_importance_df = pd.DataFrame()
    
    for fold_id, (train_index, valid_index) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        oof, test_pred, model = train_lgbm_one_fold(X_train, y_train, X_valid, y_valid, test, params, fold_id)
        
        oof_preds[valid_index] = oof
        test_preds += test_pred / CFG.n_splits
        
        # 特徴量の重要度を保存
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = X.columns
        fold_importance_df["Importance"] = model.feature_importance(importance_type="gain")  # "split"も可能
        fold_importance_df["Fold"] = fold_id
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    # スコアの計算
    score = roc_auc_score(y, oof_preds)
    print(f"Total ROC-AUC: {score:.5f}")
    
    # 特徴量の重要度をプロット
    plot_feature_importance(feature_importance_df)
    
    return oof_preds, test_preds

def plot_feature_importance(feature_importance_df):
    """特徴量の重要度を可視化する"""
    importance_mean = feature_importance_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    importance_mean[:20].plot(kind='barh', color='blue', edgecolor='k')
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importance")
    plt.show()



params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'randim_state' : CFG.random_seed,
    'num_leaves': 43,
    'max_depth' : -1,
    'learning_rate': 10**-1.47,
    'feature_fraction': 0.49,
    'bagging_fraction': 0.97,
    'bagging_freq': 10,
    'lambda_l2': 0.53,
    'verbose': -1,
    'n_estimators': 679
}



def main():
    oof, test_preds = train_lgbm(train_df, train_target, test_df, params)
    sub[CFG.target] = test_preds
    sub.to_csv(f"submission_{CFG.exp_name}.csv", index=False)
    df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "train.csv"))
    oof_df = pd.DataFrame({"id":df["id"], "oof":oof.tolist()})
    oof_df.to_csv(f"oof_{CFG.exp_name}.csv", index=False)

if __name__ == "__main__" :
    main()

