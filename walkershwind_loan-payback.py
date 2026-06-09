# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier#アンサンブル
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
train['grade_rank'] = train['grade_subgrade'].str.extract('([A-Za-z]+)', expand=False).map(grade_map)
test['grade_rank'] = test['grade_subgrade'].str.extract('([A-Za-z]+)', expand=False).map(grade_map)
train['risk_score'] = (train['debt_to_income_ratio'] * 40 + 
                       (1 - train['credit_score']/850) * 30 + train['interest_rate'] * 2)
test['risk_score'] = (test['debt_to_income_ratio'] * 40 + 
                      (1 - test['credit_score']/850) * 30 + test['interest_rate'] * 2)
train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']



print(train.head())
print(train.describe())
dummies=pd.get_dummies(train[["gender","marital_status","education_level","employment_status","loan_purpose"]])
#"education_level":['High School' "Master's" "Bachelor's" 'PhD' 'Other']
#"marital_status":['Single' 'Married' 'Divorced' 'Widowed']
#"gender":['Female' 'Male' 'Other']
#"employment_status":['Self-employed' 'Employed' 'Unemployed' 'Retired' 'Student']
#"loan_purpose":['Other' 'Debt consolidation' 'Home' 'Education' 'Vacation' 'Car' 'Medical' 'Business']
X=train[["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate",'grade_rank','risk_score','total_debt']].join(dummies)
y=train["loan_paid_back"]


X.head()


y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cuda',
    'enable_categorical': True,
}


test_dummies=pd.get_dummies(test[["gender","marital_status","education_level","employment_status","loan_purpose"]])
test_df=test[["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate",'grade_rank','risk_score','total_debt']].join(test_dummies)
print(test_df.head())
print(test_df.describe())




oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test_df.copy()

    #rf = RandomForestClassifier(n_estimators=100, random_state=1234)
    xgb = XGBClassifier(
    objective='binary:logistic',  # 目的関数
    eval_metric='auc',            # 評価指標
    max_depth=5,                  # 木の深さ
    colsample_bytree=0.8,         # 木ごとの列サンプル割合
    subsample=0.8,                # データサンプルの割合
    n_estimators=10000,           # 木の本数
    learning_rate=0.01,           # 学習率
    early_stopping_rounds=100,    # 早期停止
    random_state=42,              # 再現性のための乱数シード
    n_jobs=-1,                    # 使用するコア数
    enable_categorical=False       # カテゴリカルデータのサポート
    )
    catboost = CatBoostClassifier(random_state=1234, verbose=0,early_stopping_rounds=100, )
    
    # XGBoostの訓練に早期停止を追加
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],verbose=1000)
    
    # CatBoostの訓練に早期停止を追加
    catboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=1000)
    
    # RandomForestは早期停止をサポートしていないため、そのまま訓練
    #rf.fit(X_train, y_train)
    
    
    #val_preds_rf = rf.predict_proba(X_val)[:, 1]
    val_preds_xgb = xgb.predict_proba(X_val)[:, 1]
    val_preds_catboost = catboost.predict_proba(X_val)[:, 1]

    # 各モデルの予測結果を平均
    #val_preds = (val_preds_rf + val_preds_xgb + val_preds_catboost) / 3
    val_preds = (val_preds_catboost + val_preds_xgb) / 2
    oof_preds[val_idx] = val_preds

    # AUCスコアを計算
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')

    # テストセットで予測（アンサンブル予測の平均を取る）
    #test_preds_rf = rf.predict_proba(X_test)[:, 1]
    test_preds_xgb = xgb.predict_proba(X_test)[:, 1]
    test_preds_catboost = catboost.predict_proba(X_test)[:, 1]

    # 各モデルの予測結果を平均
    #test_preds += (test_preds_rf + test_preds_xgb + test_preds_catboost) / 3 / N_SPLITS
    test_preds += (test_preds_catboost + test_preds_xgb) / 2 / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')


output_df = pd.DataFrame({
    'id': test['id'],  # test_dfに 'id' 列があると仮定
    'loan_paid_back': test_preds  # 予測結果
})

# 予測結果をCSVファイルとして保存
output_df.to_csv('predictions_with_id_and_loan_paid_back.csv', index=False)
print("saved")

