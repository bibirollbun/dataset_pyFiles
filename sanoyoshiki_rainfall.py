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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test["winddirection"] = test["winddirection"].fillna(test["winddirection"].mode()[0])


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


features = ['pressure', "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
X = train[features]
y = train["rainfall"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 基本モデルの定義
lr = LogisticRegression()
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(tree_method='hist', device="cuda")

# グリッドサーチで調整するハイパーパラメータの範囲
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正則化パラメータ
    'solver': ['liblinear', 'saga'],  # 最適化アルゴリズム
    'max_iter': [100, 200, 300],  # 最大反復回数
    'penalty': ['l1', 'l2'],  # 正則化の種類
    'fit_intercept': [True, False]  # 切片を加えるかどうか
}


param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],  # 決定木の数
    'max_depth': [10, 20, 30, None],  # 木の深さ
    'min_samples_split': [2, 5, 10],  # 分割するための最小サンプル数
    'min_samples_leaf': [1, 2, 4],  # 葉の最小サンプル数
}


param_grid_xgb = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],  # 学習率
    'n_estimators': [50, 100, 150, 200],  # 決定木の数
    'max_depth': [3, 5, 7, 9],  # 木の深さ
    'min_child_weight': [1, 3, 5],  # 最小葉の重み
    'subsample': [0.5, 0.7, 1],  # サンプリング比率
    'colsample_bytree': [0.5, 0.7, 1],  # 木ごとの特徴量サンプリング比率
    'gamma': [0, 0.1, 0.2],  # 正則化パラメータ（木の分割に対する最小損失増加）
    'scale_pos_weight': [1, 10, 20]  # クラス不均衡に対する調整パラメータ
}


# GridSearchCVによるハイパーパラメータの最適化
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5)

# 各モデルの学習
grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)

# 最適なハイパーパラメータの表示
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)



# 最適なモデルでアンサンブル学習を作成
voting_clf = VotingClassifier(
    estimators=[
        ('lr', grid_search_lr.best_estimator_),
        ('rf', grid_search_rf.best_estimator_),
        ('xgb', grid_search_xgb.best_estimator_)
    ],
    voting='soft'  # 確率に基づく投票（硬い投票なら 'hard'）
)

# アンサンブルモデルの学習
voting_clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score

# 予測
y_pred = voting_clf.predict(X_valid)
y_pred_proba = voting_clf.predict_proba(X_valid)[:, 1]  # AUC計算用

# 精度とAUCを表示
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("AUC-ROC:", roc_auc_score(y_valid, y_pred_proba))


test_data = test[features]
X_test_scaled = scaler.transform(test_data)
y_preds = voting_clf.predict_proba(X_test_scaled)[:, 1]
data = {"id": test["id"], "rainfall": y_preds}

submission = pd.DataFrame(data)

submission.to_csv('submissionv5.csv', index=False)







