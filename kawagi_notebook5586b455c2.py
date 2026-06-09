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


import sys
print(sys.version)


# 念のため train/test/submission の読み込みをもう一度
import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')



train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
train.head()


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=train, x='Age', bins=20, kde=True)
plt.title("年齢の分布")
plt.show()



sns.countplot(data=train, x='Gender', hue='Response')
plt.title("性別と保険加入意向")
plt.show()


# 1. Annual_Premium と Response の関係を可視化（分布＋平均）
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x="Annual_Premium", hue="Response", kde=True, bins=50)
plt.title("Annual_Premium vs Response")
plt.show()


# 2. Driving_License (0/1) の比率
license_ratio = train.groupby("Driving_License")["Response"].mean()
license_ratio.plot(kind="bar", color=["skyblue", "orange"])
plt.title("Response Rate by Driving License")
plt.xlabel("Driving License (0=No, 1=Yes)")
plt.ylabel("Mean Response")
plt.show()


# 3. Region_Code ごとの Response 平均（多いので上位だけ）
region_response = train.groupby("Region_Code")["Response"].mean().sort_values(ascending=False).head(10)
region_response.plot(kind="bar", color="green")
plt.title("Top 10 Region Codes by Response Rate")
plt.xlabel("Region Code")
plt.ylabel("Mean Response")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# 数値特徴量一覧（例）
num_cols = ['Age', 'Vintage', 'Annual_Premium']

# サバイバル（ターゲット）ごとの分布確認
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=train, x=col, hue='Response', common_norm=False, fill=True)
    plt.title(f'Distribution of {col} by Response')
    plt.show()



cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Region_Code']

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=col, y='Response', data=train, ci=None)
    plt.title(f'Mean Response by {col}')
    plt.xticks(rotation=45)
    plt.show()



# KDEプロットで Response=0 と 1 の分布差を見る
for col in ['Policy_Sales_Channel', 'Previously_Insured', 'Driving_License']:
    sns.kdeplot(data=train, x=col, hue='Response', fill=True)
    plt.title(f'Distribution of {col} by Response')
    plt.show()


train.columns


# 1. ターゲットの分布
print("Response value counts:")
print(train['Response'].value_counts())
print("\nResponse割合:")
print(train['Response'].value_counts(normalize=True))

# 2. 欠損値の確認
print("\n欠損値の数（カラムごと）:")
print(train.isnull().sum())



#文字のままでは使えないカテゴリ変数を学習モデルで使用できるように変形
from sklearn.preprocessing import LabelEncoder

# Gender: Male → 1, Female → 0
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
test['Gender'] = test['Gender'].map({'Male': 1, 'Female': 0})

# Vehicle_Damage: Yes → 1, No → 0
train['Vehicle_Damage'] = train['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
test['Vehicle_Damage'] = test['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# Vehicle_Age: 順序があるのでラベルで数値化
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
train['Vehicle_Age'] = train['Vehicle_Age'].map(vehicle_age_mapping)
test['Vehicle_Age'] = test['Vehicle_Age'].map(vehicle_age_mapping)

# 変換後の確認
train[['Gender', 'Vehicle_Damage', 'Vehicle_Age']].head()



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 目的変数と特徴量
target = 'Response'
features = [col for col in train.columns if col not in ['id', target]]

# train/valid に分割（正しいコード！）
X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], train[target], test_size=0.2, random_state=42
)

# LightGBM用データセットに変換
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# ハイパーパラメータ（シンプルでOK）
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

# 学習
from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=100,
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)

# 検証用のAUCスコア
y_pred = model.predict(X_valid)
auc = roc_auc_score(y_valid, y_pred)
print(f"AUCスコア: {auc:.4f}")



print(test.columns)


# 学習データとテストデータの読み込み
train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')

# one-hot encoding（カテゴリ変数含む全て）
train_encoded = pd.get_dummies(train)
test_encoded = pd.get_dummies(test)

# 列を揃える（test に train の列がないときは補完、逆も）
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# 特徴量と目的変数
target = 'Response'
features = [col for col in train_encoded.columns if col not in ['id', target]]

X = train_encoded[features]
y = train_encoded[target]
X_test = test_encoded[features]  # test の特徴量も同じ列順に

# 5-fold交差検証
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
test_preds = np.zeros(len(test_encoded))  # テストデータ予測の平均用

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )

    # validation AUC
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

    # テストデータの予測（foldごとに平均）
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

print(f"\n✅ 平均AUC: {np.mean(auc_scores):.4f}")


# 提出ファイル作成
submission = sample_submission.copy()
submission['Response'] = test_preds
submission.to_csv('submission.csv', index=False)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ROC曲線を計算
fpr, tpr, _ = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr)

# プロット
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 閾値0.5で2値化
y_pred_binary = (y_pred > 0.5).astype(int)

# 混同行列を計算・描画
cm = confusion_matrix(y_valid, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

