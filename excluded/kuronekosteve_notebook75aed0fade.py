# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')


print('Train shape:', train.shape)
print('Test shape:', test.shape)
train.head()


print(train.info())
train.describe()


plt.figure(figsize=(6,4))
sns.countplot(x='Response', data=train)
plt.title('Response distribution')
plt.show()


train.isnull().sum()


sns.countplot(x = 'Gender', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


fig = sns.FacetGrid(train, col='Response', hue='Response', height=5)
fig.map(sns.histplot, 'Age', bins=30)


min_age = train['Age'].min()
max_age = train['Age'].max()

bins = list(range(0, int(max_age) + 10, 10))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

train['AgeBand']=pd.cut(train.Age, bins=bins, labels = labels)
train.drop('Age', axis = 1, inplace = True)
train

# age_bins =[20, 30, 40, 50, 60, 70, 80, 90]
# age_labels=['20s', '30s', '40s','50s', '60s', '70s', '80s']
# train['AgeBand']=pd.cut(train.Age, bins=age_bins, labels=age_labels, right=False)
# train.drop('Age', axis = 1, inplace = True)
# train


sns.countplot(x = 'Driving_License', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


plt.figure(figsize=(30,10))
sns.countplot(x = 'Region_Code', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


train.drop('Region_Code', axis = 1, inplace = True)
train


plt.figure(figsize=(6,4))
sns.countplot(x = 'Previously_Insured', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


sns.countplot(x = 'Vehicle_Age', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


sns.countplot(x = 'Vehicle_Damage', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


fig = sns.FacetGrid(train, col='Response', hue='Response', height=5)
fig.map(sns.histplot, 'Annual_Premium', bins=30)


min_premium = train['Annual_Premium'].min()
max_premium = train['Annual_Premium'].max()

bins = list(range(0, int(max_premium) + 100000, 100000))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

train['AnnualPremiumBand'] = pd.cut(train['Annual_Premium'], bins=bins, labels = labels)
train.drop('Annual_Premium', axis = 1, inplace = True)
train


plt.figure(figsize=(30,20))
sns.countplot(x = 'Policy_Sales_Channel', hue = 'Response', data = train)
plt.legend(loc = "upper right", title = "Gender ~ Response")


train.drop('Policy_Sales_Channel', axis = 1, inplace = True)
train


plt.figure(figsize=(6,4))
fig = sns.FacetGrid(train, col='Response', hue='Response', height=5)
fig.map(sns.histplot, 'Vintage', bins=30)


min_vintage = train['Vintage'].min()
max_vintage = train['Vintage'].max()

bins = list(range(0, int(max_vintage) + 100, 100))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

train['VintageBand'] = pd.cut(train['Vintage'], bins=bins, labels=labels)
train.drop('Vintage', axis = 1, inplace = True)
train


#性別をダミー変数
train=pd.get_dummies(train, columns=['Gender'], drop_first=True)
#車の年代も
train=pd.get_dummies(train, columns=['Vehicle_Age'], drop_first=True)
#損傷歴をYes,NoからTrue,Falseに
train['Vehicle_Damage'] = train['Vehicle_Damage'].map({'Yes': True, 'No': False})
train


# データ量が大きいため、データを標準化
scaler = StandardScaler()
train_copied = train.copy()
scaler.fit_transform(train_copied[['AgeBand', 'AnnualPremiumBand', 'VintageBand']])
train_scaled = pd.DataFrame(scaler.transform(train_copied[['AgeBand', 'AnnualPremiumBand', 'VintageBand']]))

train[['AgeBand', 'AnnualPremiumBand', 'VintageBand']] = train_scaled
train


#前処理
y = train['Response']
X = train.drop(['id', 'Response'], axis=1)

#訓練・検証データに分ける
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#モデル生成(ランダムフォレスト)
# model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
# model.fit(X_train, y_train)
# バリデーションデータで予測
# val_preds = model.predict(X_val)

# # 評価
# print('Classification Report:\n', classification_report(y_val, val_preds))
# print('Confusion Matrix:\n', confusion_matrix(y_val, val_preds))

#モデル生成(LightLgb)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
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
y_pred = model.predict(X_val)
auc = roc_auc_score(y_val, y_pred)
print(f"AUCスコア: {auc:.4f}")


#テストデータも訓練と合わせる
min_age = test['Age'].min()
max_age = test['Age'].max()
bins = list(range(0, int(max_age) + 10, 10))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
test['AgeBand']=pd.cut(test.Age, bins=bins, labels = labels)
test.drop('Age', axis = 1, inplace = True)

test.drop('Region_Code', axis = 1, inplace = True)

min_premium = test['Annual_Premium'].min()
max_premium = test['Annual_Premium'].max()
bins = list(range(0, int(max_premium) + 100000, 100000))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
test['AnnualPremiumBand'] = pd.cut(test['Annual_Premium'], bins=bins, labels = labels)
test.drop('Annual_Premium', axis = 1, inplace = True)

test.drop('Policy_Sales_Channel', axis = 1, inplace = True)

min_vintage = test['Vintage'].min()
max_vintage = test['Vintage'].max()
bins = list(range(0, int(max_vintage) + 100, 100))
labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
test['VintageBand'] = pd.cut(test['Vintage'], bins=bins, labels=labels)
test.drop('Vintage', axis = 1, inplace = True)

test=pd.get_dummies(test, columns=['Gender'], drop_first=True)
test=pd.get_dummies(test, columns=['Vehicle_Age'], drop_first=True)
test['Vehicle_Damage'] = test['Vehicle_Damage'].map({'Yes': True, 'No': False})

scaler = StandardScaler()
test_copied = test.copy()
scaler.fit_transform(test_copied[['AgeBand', 'AnnualPremiumBand', 'VintageBand']])
test_scaled = pd.DataFrame(scaler.transform(test_copied[['AgeBand', 'AnnualPremiumBand', 'VintageBand']]))
test[['AgeBand', 'AnnualPremiumBand', 'VintageBand']] = test_scaled
test


X_test = test.drop(['id'], axis=1)

test_preds = model.predict(X_test)

submission['Response'] = test_preds
submission.to_csv('submission.csv', index=False)

print('submission.csv を作成しました！')

