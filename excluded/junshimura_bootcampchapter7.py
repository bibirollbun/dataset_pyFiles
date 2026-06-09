# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# install
!pip install optuna


import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score 
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class Paths:
    P = "/kaggle/input/playground-series-s4e1/"
    train = P + "train.csv"
    test = P + "test.csv"
    sample = P + "sample_submission.csv"


# read data
train = pd.read_csv(Paths.train)


# check data
train.head()


train.shape


train.dtypes


train.isnull().sum()


train.describe()


train.nunique()


# データを分割
features = ["CreditScore", "Age"]

X_train = train[features]
y_train = train["Exited"]

print(X_train.shape)
print(y_train.shape)


y_train.value_counts()


params = {
    "boosting_type": "gbdt",  # 決定木をベースとした勾配ブースティングのアルゴリズム (Gradient Boosting Decision Tree)
    "objective": "binary",  # 二値分類 (Binary Classification) の目的関数
    "metric": "auc",  # モデルの評価指標として AUC (Area Under the Curve) を使用
    "learning_rate": 0.1,  # 学習率。各木の寄与度を調整
    "random_state": 123,  # 再現性を確保するための乱数シード
    "importance_type": "gain",  # 特徴量の重要度を計算する際の基準を「分割時の情報の利得」とする
    "verbose": -1  # ログ出力を無効にする
}


n_splits = 5
# モデルの性能評価を行うために、データを訓練用と検証用に分割するための StratifiedKFold オブジェクトを初期化しています。
# StratifiedKFold は、各分割でクラスの比率が元のデータセットと同じになるようにデータを分割します。
# shuffle=True はデータをシャッフルしてから分割することを示し、random_state は再現性を確保します。
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

metrics = []
imp = pd.DataFrame()
# cv.split() を使用して、訓練データと検証データのインデックスを取得し、n_splits で指定された回数 (ここでは 5 回) だけループ処理を行います。
for nfold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print("=" * 10, nfold, "=" * 10)
    # 訓練データのインデックス (train_idx) を使って、訓練用の特徴量 (x_tr) と目的変数 (y_tr) を抽出しています。
    x_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    # 検証データのインデックス (val_idx) を使って、検証用の特徴量 (x_va) と目的変数 (y_va) を抽出しています。
    x_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    # LightGBM 分類器モデルを初期化しています。**params は、モデルのパラメータを辞書形式で渡すための構文です。
    model = lgb.LGBMClassifier(**params)
    # 訓練データと検証データを使ってモデルを訓練しています。
    # eval_set には、訓練中に追加で評価するデータセットを指定しています。ここでは訓練データと検証データの両方を指定しています。
    # callbacks には、モデルの訓練中に実行されるコールバック関数を指定しています。
    # lgb.early_stopping は、検証データの性能が改善しなくなった場合に訓練を早期に停止します。
    # lgb.log_evaluation は、評価結果のログを出力します。ここでは、verbose=0 にしてログ出力を抑制しています。
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_tr, y_tr), (x_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(0)
        ],
    )
    # 訓練済みモデルを使って、訓練データと検証データの両方で予測確率を計算しています。
    # predict_proba() は各クラスに属する確率を返すため、[:, 1] で正例 (クラス 1) になる確率を抽出しています。
    y_tr_pred = model.predict_proba(x_tr)[:, 1]
    y_va_pred = model.predict_proba(x_va)[:, 1]
    # roc_auc_score を使って、訓練データと検証データに対する ROC-AUC スコアを計算しています。
    # ROC-AUC は、分類モデルの性能を評価するための一般的な指標です。
    metric_tr = roc_auc_score(y_tr, y_tr_pred)
    metric_va = roc_auc_score(y_va, y_va_pred)
    # 計算されたROC-AUCスコアと、現在の分割のインデックス (nfold) をリストとして metrics リストに追加しています。
    metrics.append([nfold, metric_tr, metric_va])

    # モデルの訓練で計算された特徴量の重要度を取得し、DataFrame を作成しています。
    # これにより、どの特徴量がモデルの予測に最も貢献しているかを分析できます。
    _imp = pd.DataFrame({
        "col": X_train.columns,
        "imp": model.feature_importances_,
        "nfold": nfold
    })
    # 各分割で計算された特徴量の重要度 DataFrame を imp DataFrame に連結しています。
    # axis=0 で行方向に連結し、ignore_index=True でインデックスをリセットしています。
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)



metrics_array = np.array(metrics)

print("[cv ] tr: {:.2f}+-{:.4f}, va: {:.2f}+-{:.4f}".format(
    metrics_array[:,1].mean(), metrics_array[:,1].std(),
    metrics_array[:,2].mean(), metrics_array[:,2].std(),
    
))


imp


imp_df = imp.groupby('col')['imp'].agg(['mean','std'])
imp_df.columns = ['imp', 'imp_std']

imp_df = imp_df.sort_values(by='imp', ascending=False)
imp_df


test = pd.read_csv(Paths.test)
test.shape


X_test = test[features]
X_test.shape


submit = pd.read_csv(Paths.sample)
submit.head(10)


y_test_pred = model.predict_proba(X_test)[:,1]
df_submit = pd.DataFrame({
    'id': test['id'],
    'Exited': y_test_pred
    
})
df_submit


df_submit.to_csv('submission.csv',index=False)

