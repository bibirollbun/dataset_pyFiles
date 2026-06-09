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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class CFG :
    debug = False
    random_seed = 42
    n_splits = 5
    num_classes = 2
    target = "Exited"
    drop_col = ["id", "CustomerId", "Surname"]
    cat_col = ["Geography", "Gender"]
    data_dir = "../input/"
    exp_name = "lgbm_baseline"


class lgbm_CFG : 
    # ここらへんは変えない方がいいかも？
    print_freq = 100 # 何ラウンドごとに評価するか
    num_rounds = 10000 # 何ラウンド回すか、大きめの値を入れておいて早期終了する
    early_stopping_rounds = 100 # 何ラウンド改善がなかったら終了するか
    metric = "auc" # 評価指標
    objective = "binary" # 二値分類なので
    boosting = "gbdt" # 今回は gbdt というブースティング手法を使います
    
    # ここからはデフォルトの値を入れてます！
    learning_rate = 0.1
    max_depth = -1
    num_leaves = 31
    min_data_in_leaf = 20
    bagging_fraction = 1.0
    bagging_freq = 0
    feature_fraction = 1.0
    lambda_l1 = 0.0
    lambda_l2 = 0.0



train_df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "test.csv"))
sub = pd.read_csv(os.path.join(CFG.data_dir, "playground-series-s4e1", "sample_submission.csv"))
train_df.head()


# EDA
print(train_df.shape)
print(test_df.shape)


# 欠損値の確認
print(train_df.isnull().sum())


# CreditScore の分布    
plt.figure(figsize=(8, 4))
plt.hist(train_df["CreditScore"], bins=30)
plt.show()


# Age の分布
plt.figure(figsize=(8, 4))
plt.hist(train_df["Age"], bins=30)
plt.show()



# Tenure の分布、棒グラフで表示
plt.figure(figsize=(8, 4))
train_df["Tenure"].value_counts().sort_index().plot(kind="bar")
plt.show()



# Balance の分布
plt.figure(figsize=(8, 4))
plt.hist(train_df["Balance"], bins=30)
plt.show()



# NumOfProducts の分布、棒グラフで表示、横軸は数字の大きさ順に並べる
plt.figure(figsize=(8, 4))
train_df["NumOfProducts"].value_counts().sort_index().plot(kind="bar")
plt.show()


def process_data(df, is_train=True) :
    df = df.drop(CFG.drop_col, axis=1)
    for col in CFG.cat_col :
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # ここに特徴量エンジニアリングを追加する。ここでは例として is_balance_zero を追加している
    df["is_balance_zero"] = (df["Balance"] == 0).astype(int)
    
    if is_train :
        return df.drop(CFG.target, axis=1), df[CFG.target]
    else :
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


def train_lgbm(X, y, test, params) :
    # oof の結果も返す
    oof_preds = np.zeros((X.shape[0], ))
    test_preds = np.zeros((test.shape[0], ))
    
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    for fold_id, (train_index, valid_index) in enumerate(skf.split(X, y)) :
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        oof, test_preds, _ = train_lgbm_one_fold(X_train, y_train, X_valid, y_valid, test, params, fold_id)
        
        oof_preds[valid_index] = oof
        test_preds += test_preds / CFG.n_splits
    
    score = roc_auc_score(y, oof_preds)
    print(f"total roc-auc: {score}")
    
    return oof, test_preds


# ここでは、 lightgbm に渡すパラメータを定義
params = {
    "objective" : "binary",
    "metric" : lgbm_CFG.metric,
    "verbosity" : -1,
    "boosting_type" : lgbm_CFG.boosting,
    # "device_type" : "gpu", # 結果の一意性が保証されないためコメントアウト
    "random_state" : CFG.random_seed,
    # CFG デフォルト値につき以下省略。書きたい場合は上の書き方を参考にして記入してください。
}


def main() :
    oof, test_preds = train_lgbm(train_df, train_target, test_df, params)
    sub[CFG.target] = test_preds
    sub.to_csv(f"submission_{CFG.exp_name}.csv", index=False)
    
if __name__ == "__main__" :
    main()

