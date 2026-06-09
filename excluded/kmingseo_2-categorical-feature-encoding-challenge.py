import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
train_df.head()


test_df = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
test_df.head()
#target이 없음


print(train_df.shape, test_df.shape)


#target값 분포 확인하기
sns.countplot(data=train_df, x='target')


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

sns.countplot(data=train_df, x='bin_0', ax=axs[0, 0])
sns.countplot(data=train_df, x='bin_1', ax=axs[0, 1])
sns.countplot(data=train_df, x='bin_2', ax=axs[0, 2])
sns.countplot(data=train_df, x='bin_3', ax=axs[1, 0])
sns.countplot(data=train_df, x='bin_4', ax=axs[1, 1])

axs[1, 2].axis('off')


#범주형
for i in range(10):
    column = f'nom_{i}'
    unique = train_df[column].unique()
    print(unique)


for i in range(10):
    colunm = f'nom_{i}'
    count = train_df[colunm].nunique()
    print(count)


fig, axs = plt.subplots(2, 3, figsize=(20, 10))

sns.countplot(data=train_df, x='nom_0', ax=axs[0, 0])
sns.countplot(data=train_df, x='nom_1', ax=axs[0, 1])
sns.countplot(data=train_df, x='nom_2', ax=axs[0, 2])
sns.countplot(data=train_df, x='nom_3', ax=axs[1, 0])
sns.countplot(data=train_df, x='nom_4', ax=axs[1, 1])

axs[1, 2].axis('off')


fig, axs = plt.subplots(3, 2, figsize=(10, 10))

sns.countplot(data=train_df, x='nom_0', hue='target', ax=axs[0, 0])
sns.countplot(data=train_df, x='nom_1', hue='target', ax=axs[0, 1])
sns.countplot(data=train_df, x='nom_2', hue='target', ax=axs[1, 0])
sns.countplot(data=train_df, x='nom_3', hue='target', ax=axs[1, 1])
sns.countplot(data=train_df, x='nom_4', hue='target',ax=axs[2, 0])

axs[2, 1].axis('off')


#순서형 
for i in range(6):
    column = f'ord_{i}'
    unique = train_df[column].unique()
    print(unique)


for i in range(6):
    column = f'ord_{i}'
    count = train_df[column].nunique()
    print(count)
    


#ord_1,2 순서 지정하기
from pandas.api.types import CategoricalDtype

ord_1 = ['Grandmaster', 'Expert', 'Novice', 'Contributor', 'Master']
ord_2 = ['Cold', 'Hot', 'Lava Hot', 'Boiling Hot', 'Freezing', 'Warm']

ord_1_new = CategoricalDtype(categories = ord_1, ordered = True)
ord_2_new = CategoricalDtype(categories = ord_2, ordered = True)


train_df['ord_1'] = train_df['ord_1'].astype(ord_1_new)
train_df['ord_2'] = train_df['ord_2'].astype(ord_2_new)


fig, axs = plt.subplots(1, 3, figsize=(20, 10))

sns.countplot(data=train_df, x='ord_0', hue='target', ax = axs[0])
sns.countplot(data=train_df, x='ord_1', hue='target', ax = axs[1])
sns.countplot(data=train_df, x='ord_2', hue='target', ax = axs[2])


sns.countplot(data=train_df, x='ord_4')


sns.countplot(data=train_df, x='day',hue='target')


sns.countplot(data=train_df, x='month', hue='target')


data = pd.concat([train_df, test_df], ignore_index = True)
data.drop(columns = 'target', axis = 1, inplace=True)

train_df_target = train_df['target']


#bin_3, bin_4 인코딩하기 (문자열을 숫자로 바꾸기)
data['bin_3'] = data['bin_3'].apply(lambda x: 1 if x=='T' else 0)
data['bin_4'] = data['bin_4'].apply(lambda x: 1 if x=='Y' else 0)

bin_ = [f'bin_{i}' for i in range(0, 5)]
bin_data = data[bin_]
bin_data.head()


#순서 없는 변수들 인코딩
nom_ = [f'nom_{i}' for i in range(10)]
nom_data = data[nom_]
list_ = ['day', 'month']
nom_.extend(list_)
nom_data = data[nom_]

nom_data.head()


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
nom_enc = encoder.fit_transform(nom_data)

nom_enc


#순서 변수 인코딩
ord_ = [f'ord_{i}' for i in range(6)]
ord_data = data[ord_]

ord1 = {'Grandmaster': 4 , 'Expert': 2, 'Novice': 0, 'Contributor': 1, 'Master': 3}
ord2 = {'Cold': 1, 'Hot': 3, 'Lava Hot': 5, 'Boiling Hot': 4, 'Freezing': 0, 'Warm': 2}

ord_data['ord_1'] = ord_data['ord_1'].map(ord1).fillna(-1).astype(int)
ord_data['ord_2'] = ord_data['ord_2'].map(ord2).fillna(-1).astype(int)

ord_data.head()
#print(ord_data.dtypes)


#ord_3, 4, 5 labelencoder로 부여하기
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

ord_data['ord_3'] = encoder.fit_transform(ord_data['ord_3'].values)
ord_data['ord_4'] = encoder.fit_transform(ord_data['ord_4'].values)
ord_data['ord_5'] = encoder.fit_transform(ord_data['ord_5'].values)

ord_data.head()


new_data = pd.concat([bin_data, pd.DataFrame(ord_data, columns = ord_data.columns)], axis = 1)

new_data.dtypes


from scipy import sparse

final_data = sparse.hstack([sparse.csr_matrix(new_data), nom_enc],
                             format = 'csr')

final_data


train_new_data = final_data[:len(train_df), :]
test_new_data = final_data[len(train_df):, :]
print(train_new_data.shape, test_new_data.shape)


from sklearn.model_selection import train_test_split


sub_input, val_input, sub_target, val_target = train_test_split(
    train_new_data, train_df_target, stratify = train_df_target, random_state = 42
)

print(sub_input.shape)
print(val_input.shape)


#from sklearn.ensemble import RandomForestClassifier

#rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, max_depth)
#rf.fit(sub_input, sub_target)
#print(rf.score(sub_input, sub_target))


from sklearn.linear_model import LogisticRegression

#C값을 낮춰서 정규화 강도 높임
lr = LogisticRegression(C=0.1, max_iter=1000)
lr.fit(sub_input, sub_target)

print(lr.score(sub_input, sub_target))


print(lr.score(val_input, val_target))


y_pred = lr.predict_proba(val_input)[:, 1]

from sklearn.metrics import roc_auc_score
roc_auc_score(val_target, y_pred)


submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

y_pred = lr.predict_proba(test_new_data)
#del submission['target'] #기존 열 삭제 
submission['target'] = np.nan
submission['target'] = y_pred[:, 1]

submission.to_csv('sub.csv', index = False)

