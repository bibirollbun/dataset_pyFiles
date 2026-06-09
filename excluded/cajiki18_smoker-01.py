# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    print(dirname)
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd

# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/playground-series-s3e24/train.csv')
test = pd.read_csv('../input/playground-series-s3e24/test.csv')
original=pd.read_csv("../input/smoker-status-prediction-using-biosignals/train_dataset.csv")

# print(train)
# print(test)
# print(original)


# Idを除外する
train_copy=train.copy()
test_copy=test.copy()
original_copy=original.copy()

train = train.drop(['id'], axis=1)

train=pd.concat([train,original],axis=0)
train.head()


train_x = train.drop(['smoking'], axis=1) # ALL特徴量
train_y = train['smoking']
test_x = test.copy()


# 例: 身長(cm) の場合
# print("\n--- 身長(cm) の統計量（非喫煙者 vs 喫煙者）---")
# print("非喫煙者 (smoking=0) - 身長(cm) の平均:", train[train['smoking'] == 0]['height(cm)'].mean())
# print("喫煙者 (smoking=1) - 身長(cm) の平均:", train[train['smoking'] == 1]['height(cm)'].mean())
# print("\n非喫煙者 (smoking=0) - 身長(cm) の要約統計量:\n", train[train['smoking'] == 0]['height(cm)'].describe())
# print("\n喫煙者 (smoking=1) - 身長(cm) の要約統計量:\n", train[train['smoking'] == 1]['height(cm)'].describe())


print(train_x)


###clipping###

p01 = train_x['Gtp'].quantile(0.01)
p99 = train_x['Gtp'].quantile(0.99)
print(p01, p99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x['Gtp'] = train_x['Gtp'].clip(p01, p99)
test_x['Gtp'] = test_x['Gtp'].clip(p01, p99)

p01 = train_x['AST'].quantile(0.01)
p99 = train_x['AST'].quantile(0.99)
print(p01, p99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x['AST'] = train_x['AST'].clip(p01, p99)
test_x['AST'] = test_x['AST'].clip(p01, p99)

p01 = train_x['ALT'].quantile(0.01)
p99 = train_x['ALT'].quantile(0.99)
print(p01, p99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x['ALT'] = train_x['ALT'].clip(p01, p99)
test_x['ALT'] = test_x['ALT'].clip(p01, p99)

p01 = train_x['LDL'].quantile(0.01)
p99 = train_x['LDL'].quantile(0.99)
print(p01, p99)

# 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
train_x['LDL'] = train_x['LDL'].clip(p01, p99)
test_x['LDL'] = test_x['LDL'].clip(p01, p99)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# グラフを描画する特徴量のリストを作成
# 'id' と 'smoking' 以外の全ての列を対象とする
features_to_plot = train.columns.drop(['smoking']).tolist()

# グラフのスタイル設定
sns.set_style("whitegrid")

# 各特徴量に対してヒストグラムを描画（smoking=0 と smoking=1 を重ねて表示）
for feat in features_to_plot:
    plt.figure(figsize=(10, 6)) # 図のサイズを調整

    sns.histplot(data=train, x=feat, hue='smoking',
                 bins=50, # ビンの数
                 kde=True, # カーネル密度推定（滑らかな曲線）も表示
                 palette={0: 'skyblue', 1: 'lightcoral'}, # smoking=0は水色、smoking=1は赤系
                 common_norm=False, # 各グループ（smoking=0と1）でそれぞれ正規化
                 alpha=0.6) # 透明度を設定して重なりを見やすくする

    plt.title(f'{feat} Distribution by Smoker Status (Train Data)', fontsize=16)
    plt.xlabel(feat, fontsize=12)
    plt.ylabel('Frequency (Normalized)', fontsize=12) # common_norm=False の場合、正規化された頻度
    # plt.legend(title='Smoking Status', labels=['Non-Smoker (0)', 'Smoker (1)']) # <--- この行を削除またはコメントアウト！

    plt.tight_layout() # サブプロット間の余白を自動調整
    plt.show()



###特徴量作成###

train_x['BMI'] = (train_x['weight(kg)'] / (train_x['height(cm)'] / 100)**2)
test_x['BMI'] = (test_x['weight(kg)'] / (test_x['height(cm)'] / 100)**2)

train_x['triglyceride_over_150'] = (train_x['triglyceride'] >= 150).astype(int)
test_x['triglyceride_over_150'] = (test_x['triglyceride'] >= 150).astype(int)

train_x['HDL-LDL_Ratio'] = (train_x['HDL'] / train_x['LDL'])
test_x['HDL-LDL_Ratio'] = (test_x['HDL'] / test_x['LDL'])

train_x['AST-ALT_Ratio'] = (train_x['AST'] / train_x['ALT'])
test_x['AST-ALT_Ratio'] = (test_x['AST'] / test_x['ALT'])


# 新しい特徴量 HW_Ratio を追加
# 身長(height(cm)) と ウエスト周囲径(waist(cm)) の比率
train_x['HW_Ratio'] = train_x['height(cm)'] / train_x['waist(cm)']
test_x['HW_Ratio'] = test_x['height(cm)'] / test_x['waist(cm)']

# 新しい特徴量 HA_Ratio を追加
# 身長(height(cm)) と 年齢(age) の比率
train_x['HA_Ratio'] = train_x['height(cm)'] / train_x['age']
test_x['HA_Ratio'] = test_x['height(cm)'] / test_x['age']

train_x['HS_Ratio'] = train_x['height(cm)'] / train_x['systolic']
test_x['HS_Ratio'] = test_x['height(cm)'] / test_x['systolic']

train_x['HR_Ratio'] = train_x['height(cm)'] / train_x['relaxation']
test_x['HR_Ratio'] = test_x['height(cm)'] / test_x['relaxation']

# --- 'hearing' の加工 ---
# 左右の聴力の良い方 (小さい値が良い) を計算し、'hearing(left)' に上書き
best_hearing_train = np.where(train_x['hearing(left)'] < train_x['hearing(right)'],
                              train_x['hearing(left)'],  train_x['hearing(right)'])
worst_hearing_train = np.where(train_x['hearing(left)'] < train_x['hearing(right)'],
                               train_x['hearing(right)'],  train_x['hearing(left)'])

train_x['hearing(left)'] = best_hearing_train - 1
train_x['hearing(right)'] = worst_hearing_train - 1

best_hearing_test = np.where(test_x['hearing(left)'] < test_x['hearing(right)'],
                             test_x['hearing(left)'],  test_x['hearing(right)'])
worst_hearing_test = np.where(test_x['hearing(left)'] < test_x['hearing(right)'],
                              test_x['hearing(right)'],  test_x['hearing(left)'])

test_x['hearing(left)'] = best_hearing_test - 1
test_x['hearing(right)'] = worst_hearing_test - 1


# --- 'eyesight' の加工 ---
# 異常に大きい視力値（9より大きい）を0にクリッピング（欠損値扱いまたは異常値）
train_x['eyesight(left)'] = np.where(train_x['eyesight(left)'] > 9, 0, train_x['eyesight(left)'])
train_x['eyesight(right)'] = np.where(train_x['eyesight(right)'] > 9, 0, train_x['eyesight(right)'])

test_x['eyesight(left)'] = np.where(test_x['eyesight(left)'] > 9, 0, test_x['eyesight(left)'])
test_x['eyesight(right)'] = np.where(test_x['eyesight(right)'] > 9, 0, test_x['eyesight(right)'])

best_eyesight_train = np.where(train_x['eyesight(left)'] > train_x['eyesight(right)'],
                               train_x['eyesight(left)'],  train_x['eyesight(right)'])
worst_eyesight_train = np.where(train_x['eyesight(left)'] > train_x['eyesight(right)'],
                                train_x['eyesight(right)'],  train_x['eyesight(left)'])

train_x['eyesight(left)'] = best_eyesight_train
train_x['eyesight(right)'] = worst_eyesight_train

best_eyesight_test = np.where(test_x['eyesight(left)'] > test_x['eyesight(right)'],
                              test_x['eyesight(left)'],  test_x['eyesight(right)'])
worst_eyesight_test = np.where(test_x['eyesight(left)'] > test_x['eyesight(right)'],
                               test_x['eyesight(right)'],  test_x['eyesight(left)'])

test_x['eyesight(left)'] = best_eyesight_test
test_x['eyesight(right)'] = worst_eyesight_test


# ---------------------------------
# xgboostの特徴量の重要度
# ---------------------------------

import xgboost as xgb

# xgboost
dtrain = xgb.DMatrix(train_x, label=train_y)
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50
model = xgb.train(params, dtrain, num_round)

# 重要度の上位を出力する
fscore = model.get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
for rank, (col, imp) in enumerate(fscore[:300]):
    print(f"Top {rank+1}: {imp} {col}")


selected_features = [
    'height(cm)',
    'hemoglobin',
    'Gtp',
    'triglyceride',
    'HA_Ratio',
    'LDL',
    'serum creatinine',
    'ALT',
    'AST',
    'Cholesterol',
    'BMI',
    'age',
    'HDL',
    'fasting blood sugar',
    'dental caries',
    'HS_Ratio',
    'relaxation',
    'AST-ALT_Ratio',
    'HR_Ratio',
    'waist(cm)',
    'systolic',
    'HW_Ratio',
    'eyesight(right)',
    'HDL-LDL_Ratio',
    # 'weight(kg)',
    # 'eyesight(left)',
    # 'hearing(right)'
    # 'Urine protein',
    # 'hearing(left)'
]


train_x = train_x[selected_features].copy()
test_x = test_x[selected_features].copy()

train_y = train_y.copy()


# print(train_x)
# print(test_x)
# print(train_y)


from sklearn.preprocessing import LabelEncoder

# 欠損補完

# train_xの欠損値を確認
print("Trainデータにおける各列の欠損値の数:")
print(train_x.isnull().sum())

# test_xの欠損値を確認
print("\nTestデータにおける各列の欠損値の数:")
print(test_x.isnull().sum())



# カテゴリ型変数はなし。Label Encodingなし。

print(train_x)
print(train_y)
print(test_x)


# 学習データを学習データとバリデーションデータに分ける
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]


# -----------------------------------
# xgboostの実装
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

# 特徴量と目的変数をxgboostのデータ構造に変換する
dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

params = {
    'booster': 'gbtree', 
    'objective': 'binary:logistic', 
    'eval_metric': 'logloss', 
    'random_state': 71
}
num_round = 500

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20)


# アーリーストッピングの結果（最適な反復回数）を表示
print(model.best_iteration)

# 最適な決定木の本数で予測を行う（※xgboost ver 1.4.0以降の書き方）
irange = (0, model.best_iteration + 1)
pred = model.predict(dtest, iteration_range=irange)
print(pred)


# バリデーションデータでのスコアの確認 (loglossとAUC)


va_pred_proba = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1)) # 最適なiterationで予測
score_logloss = log_loss(va_y, va_pred_proba)
score_auc = roc_auc_score(va_y, va_pred_proba)
print(f'Validation logloss: {score_logloss:.4f}')
print(f'Validation AUC: {score_auc:.4f}')



# 予測（二値の予測値ではなく、1である確率を出力するようにしている）
pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

# -----------------------------------
# 提出ファイルの作成
# -----------------------------------
submission = pd.DataFrame({'id': test['id'], 'smoking': pred_proba}) # 元のtest dfからidを取得
submission.to_csv('submission.csv', index=False)

