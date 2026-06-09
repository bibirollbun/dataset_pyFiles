import numpy as np
import pandas as pd

# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/playground-series-s3e24/train.csv')
test = pd.read_csv('../input/playground-series-s3e24/test.csv')


train_x = train.drop(['smoking'], axis=1)
train_y = train['smoking']

test_x = test.copy()

print(train)
#print(test)


#特徴量作成

#idを削除
train_x = train_x.drop(['id'], axis=1)
test_x = test_x.drop(['id'], axis=1)

#聴力、視力差の追加
train_x['eyesight_diff'] = (train_x['eyesight(left)'] - train_x['eyesight(right)'])
test_x['eyesight_diff'] = (test_x['eyesight(left)'] - test_x['eyesight(right)'])

train_x['hearing_diff'] = (train_x['hearing(left)'] - train_x['hearing(right)'])
test_x['hearing_diff'] = (test_x['hearing(left)'] - test_x['hearing(right)'])

#視力、聴力の削除
train_x = train_x.drop(['eyesight(left)'], axis=1)
test_x = test_x.drop(['eyesight(left)'], axis=1)
train_x = train_x.drop(['eyesight(right)'], axis=1)
test_x = test_x.drop(['eyesight(right)'], axis=1)

train_x = train_x.drop(['hearing(left)'], axis=1)
test_x = test_x.drop(['hearing(left)'], axis=1)
train_x = train_x.drop(['hearing(right)'], axis=1)
test_x = test_x.drop(['hearing(right)'], axis=1)

#BMIの作成
train_x['BMI'] = train_x['weight(kg)'] / ((train_x['height(cm)'] / 100) ** 2)
test_x['BMI'] = test_x['weight(kg)'] / ((test_x['height(cm)'] / 100) ** 2)

#ウエスト身長比
train_x['waist_height_ratio'] = train_x['waist(cm)'] / train_x['height(cm)']
test_x['waist_height_ratio'] = test_x['waist(cm)'] / test_x['height(cm)']

#肝臓系酵素の合計
#train_x['liver_sum'] = train_x['AST'] + train_x['ALT'] + train_x['Gtp']
#test_x['liver_sum'] = test_x['AST'] + test_x['ALT'] + test_x['Gtp']

#HDL/LDL 比
#train_x["hdl_ldl_ratio"] = train_x["HDL"] / (train_x["LDL"] + 1e-5)
#test_x["hdl_ldl_ratio"] = test_x["HDL"] / (test_x["LDL"] + 1e-5)

#尿蛋白の削除
train_x = train_x.drop(['Urine protein'], axis=1)
test_x = test_x.drop(['Urine protein'], axis=1)

#虫歯の削除
#train_x = train_x.drop(['dental caries'], axis=1)
#test_x = test_x.drop(['dental caries'], axis=1)

#トリグリセリドのbin化
bin_edge = [-float('inf'), 150.0, 200.0, 500.0, float('inf')]
labels = [0,1,2,3]
train_x['triglyceride_bin'] = pd.cut(train_x['triglyceride'],bins=bin_edge, labels=labels).astype(int)
test_x['triglyceride_bin'] = pd.cut(test_x['triglyceride'],bins=bin_edge, labels=labels).astype(int)

#心拍差の追加
train_x['PulsePressure'] = train_x['systolic'] - train_x['relaxation']
test_x['PulsePressure'] = test_x['systolic'] - test_x['relaxation']






# モデル
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_lgb = LGBMClassifier(random_state=71)

#クロスバリデーション
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)
scores_xgb = cross_val_score(model_xgb, train_x, train_y, cv=cross_val, scoring='roc_auc')

print(f"CV ROC AUC: {scores_xgb.mean():.4f}")
for i, score in enumerate(scores_xgb, 1):
    print(f" Fold {i}: {score:.4f}")

scores_lgb = cross_val_score(model_lgb, train_x, train_y, cv=cross_val, scoring='roc_auc')

print(f"CV ROC AUC: {scores_lgb.mean():.4f}")
for i, score in enumerate(scores_lgb, 1):
    print(f" Fold {i}: {score:.4f}")

model_xgb.fit(train_x, train_y)
model_lgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]
pred_lgb = model_lgb.predict_proba(test_x)[:, 1]

ensemble_pred = (0.2*pred_xgb + 0.8*pred_lgb)


#提出用ファイルの作成
submission = pd.DataFrame({'id': test['id'], 'smoking': ensemble_pred})
submission.to_csv('submission.csv', index=False)
submission.head()

