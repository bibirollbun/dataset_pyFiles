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


import os

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import NullFormatter

import numpy as np
import pandas as pd
import seaborn as sns

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as rf
import lightgbm as lgb
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from prophet import Prophet
from tqdm.notebook import tqdm
import time
from scipy import stats

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import torch
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchinfo import summary
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Train data has 13 features and 2190 records in all.
# Test data has 12 features and 730 records in all.

print(train_df.shape, test_df.shape)


train_df.head(3)


test_df.head(3)


print(train_df.isnull().sum())
print(test_df.isnull().sum())


train_df.iloc[:,1:].describe().T


test_df.iloc[:,1:].describe().T


# train data 
# split dataframe by each year

for i, split_end in enumerate(range(365, 2191, 365)):
    
    split_start = split_end - 365

    if i == 0:
        df_1 =  train_df.iloc[split_start:split_end, :]
    elif i == 1:
        df_2 =  train_df.iloc[split_start:split_end, :]
    elif i == 2:
        df_3 =  train_df.iloc[split_start:split_end, :]
    elif i == 3:
        df_4 =  train_df.iloc[split_start:split_end, :]
    elif i == 4:
        df_5 =  train_df.iloc[split_start:split_end, :]
    elif i == 5:
        df_6 =  train_df.iloc[split_start:split_end, :]

print(df_1.shape, df_2.shape, df_3.shape, df_4.shape, df_5.shape, df_6.shape)


# find duplicate values by the year
# train data

day_count_1 = df_1.groupby('day')['day'].agg('count').to_frame()
day_count_1.columns = ['count']

day_count_2 = df_2.groupby('day')['day'].agg('count').to_frame()
day_count_2.columns = ['count']

day_count_3 = df_3.groupby('day')['day'].agg('count').to_frame()
day_count_3.columns = ['count']

day_count_4 = df_4.groupby('day')['day'].agg('count').to_frame()
day_count_4.columns = ['count']

day_count_5 = df_5.groupby('day')['day'].agg('count').to_frame()
day_count_5.columns = ['count']

day_count_6 = df_6.groupby('day')['day'].agg('count').to_frame()
day_count_6.columns = ['count']

print('df_1', '\n', 'day:1 ~ 365 duplicate count', '\n', day_count_1.query('count != 1').count())
print('df_2', '\n', 'day:366 ~ 729 duplicate count', '\n', day_count_2.query('count != 1').count())
print('df_3', '\n', 'day:730 ~ 1094 duplicate count', '\n', day_count_3.query('count != 1').count())
print('df_4', '\n', 'day:1095 ~ 1459 duplicate count', '\n', day_count_4.query('count != 1').count())
print('df_5', '\n', 'day:1060 ~ 1824 duplicate count', '\n', day_count_5.query('count != 1').count())
print('df_6', '\n', 'day:1825 ~ 2190 duplicate count', '\n', day_count_6.query('count != 1').count())


# find duplicate values by the year
# test data

for i, split_end in enumerate(range(365, 731, 365)):
    
    split_start = split_end - 365

    if i == 0:
        df_7 =  test_df.iloc[split_start:split_end, :]
    elif i == 1:
        df_8 =  test_df.iloc[split_start:split_end, :]
        
day_count_7 = df_7.groupby('day')['day'].agg('count').to_frame()
day_count_7.columns = ['count']
day_count_8 = df_8.groupby('day')['day'].agg('count').to_frame()
day_count_8.columns = ['count']

print('df_7', '\n', 'day:1 ~ 365 duplicate count', '\n', day_count_7.query('count != 1').count())
print('df_8', '\n', 'day:366 ~ 730 duplicate count', '\n', day_count_8.query('count != 1').count())


# find duplicated values 'day'

duplicate_id = []

df_3_duplication = day_count_3.query('count != 1').index
print('duplicate day:', df_3_duplication, '\n')

for i in df_3_duplication:
    id = df_3.loc[df_3.day == i, 'id'].values
    duplicate_id.append(id)

df_4_duplication = day_count_4.query('count != 1').index
print('duplicate day:', df_4_duplication, '\n')

for i in df_4_duplication:
    id = df_4.loc[df_4.day == i, 'id'].values
    duplicate_id.append(id)

duplicate_id = np.array(duplicate_id)
print('duplicate count:', len(duplicate_id), '\n')


# check all train data
not_match_id = []

day_count_all = train_df.groupby('day')['day'].agg('count').to_frame()
day_count_all.columns = ['count']

print(day_count_all.query('count != 6').count(), '\n')

not_match_all = day_count_all.query('count != 6').index
print('not match day:', not_match_all, '\n')

for i in not_match_all:
    id = train_df.loc[train_df.day == i, 'id'].values
    not_match_id.append(id)

print('not match id count:', len(not_match_id))


# correct the columns of day

train = train_df.copy()

day_of_year = np.arange(1, 366, 1) 
df = pd.DataFrame({'day': day_of_year})

train.day = pd.concat([df, df, df, df, df, df]).values

df = train.groupby('day')['day'].agg('count').to_frame()
df.columns = ['count']

print('Check: ', df.query('count != 6').count())


# rename column's name: 'day' -> 'day of the year'

test = test_df.copy()

train = train.rename(columns={'day': 'day_of_year'})
test = test.rename(columns={'day': 'day_of_year'})

print(train.shape, test.shape)


# add new features: 'diff_temp', 'diff_maxmin'

train['diff_temp'] = train['temparature'].diff()
train['diff_temp'].fillna(0, inplace=True)

test['diff_temp'] = test['temparature'].diff()
test['diff_temp'].fillna(0, inplace=True)

train['diff_maxmin'] = train['maxtemp'] - train['mintemp']
test['diff_maxmin'] = test['maxtemp'] - test['mintemp']


fig, ax = plt.subplots(nrows=5, ncols=6,figsize=(12, 20))

for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(2, 7):
        col = train.columns[j]
        
        sns.boxplot(
            data=train.iloc[start:end, :],
            x='rainfall',
            y=col,
            hue='rainfall',
            ax=ax[j-2][i]
        )
        
        if j == 2:
            ax[j-2][i].set_ylim(990, 1040)
        if j in [3,4,5]:
            ax[j-2][i].set_ylim(3.0, 37.0)
            
        ax[j-2][i].legend().remove()
        ax[j-2][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part train data> pressure, maxtemp, temparature, mintemp, dewpoint', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(12, 5))

for j in range(2, 7):
    col = train.columns[j]
    
    sns.boxplot(
        data=train,
        x='rainfall',
        y=col,
        hue='rainfall',
        ax=ax[j-2]
    )
    
    if j == 2:
        ax[j-2].set_ylim(990, 1040)
    if j in [3,4,5]:
        ax[j-2].set_ylim(3.0, 37.0)
        
    ax[j-2].legend().remove()
    ax[j-2].set_title(f'columns : {col}')
    
plt.suptitle('<all train data> pressure, maxtemp, temparature, mintemp, dewpoint', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=2,figsize=(8, 10))

for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(2, 7):
        col = train.columns[j]
        
        sns.boxplot(
            data=train.iloc[start:end, :],
            y=col,
            ax=ax[j-2][i]
        )
 
        if j in [3,4,5]:
            ax[j-2][i].set_ylim(3.0, 37.0)

        ax[j-2][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<test data> pressure, maxtemp, temparature, mintemp, dewpoint', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=6,figsize=(12, 20))

for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(7, 12):
        col = train.columns[j]
        
        sns.boxplot(
            data=train.iloc[start:end, :],
            x='rainfall',
            y=col,
            hue='rainfall',
            ax=ax[j-7][i]
        )
        if j == 7:
            ax[j-7][i].set_ylim(35.0, 100.0)

        elif j == 8:
            ax[j-7][i].set_ylim(0.0, 105.0)

        elif j == 9:
            ax[j-7][i].set_ylim(-0.5, 12.5)
            
        elif j == 10:
            ax[j-7][i].set_ylim(-0.5, 310.0)

        elif j == 11:
            ax[j-7][i].set_ylim(0.0, 65.0)
            
        ax[j-7][i].legend().remove()
        ax[j-7][i].set_title(f'part : {part} ~ {end}')
        
plt.suptitle('<train data> humidity, cloud, sunshine, winddirection, windspeed', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(12, 5))


for j in range(7, 12):
    col = train.columns[j]
    
    sns.boxplot(
        data=train,
        x='rainfall',
        y=col,
        hue='rainfall',
        ax=ax[j-7]
    )
    
    ax[j-7].legend().remove()
    ax[j-7].set_title(f'columns : {col}')
    
plt.suptitle('<all train data> humidity, cloud, sunshine, winddirection, windspeed', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=2,figsize=(8, 10))

for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(7, 12):
        col = train.columns[j]
        
        sns.boxplot(
            data=train.iloc[start:end, :],
            y=col,
            ax=ax[j-7][i]
        )
        if j == 11:
            ax[j-7][i].set_ylim(3.0, 65.0)
        ax[j-7][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<test data> humidity, cloud, sunshine, winddirection, windspeed', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=6,figsize=(12, 8))

for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(13, 15):
        col = train.columns[j]
        
        sns.boxplot(
            data=train.iloc[start:end, :],
            x='rainfall',
            y=col,
            hue='rainfall',
            ax=ax[j-13][i]
        )
 
        if j == 13:
            ax[j-13][i].set_ylim(-12.0, 11.0)

        elif j == 14:
            ax[j-13][i].set_ylim(-1.0, 11.0)
      
        ax[j-13][i].legend().remove()
        ax[j-13][i].set_title(f'part : {part} ~ {end}')
        
plt.suptitle('<train data> diff_temp, diff_maxmin', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 5))

for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(12, 14):
        col = test.columns[j]
        
        sns.boxplot(
            data=test.iloc[start:end, :],
            y=col,
            ax=ax[j-12][i]
        )
 
        if j == 12:
            ax[j-12][i].set_ylim(-12.0, 11.0)

        elif j == 13:
            ax[j-12][i].set_ylim(-1.0, 11.0)
      
        ax[j-12][i].set_title(f'part : {part} ~ {end}')
        
plt.suptitle('<test data> diff_temp, diff_maxmin', y=1.0)
plt.tight_layout()
plt.show()


def draw_features(data, x=None, y=None, overlap=None):
    
    sns.set(font_scale=2.0)
    fig, ax = plt.subplots(figsize=(30,8))
    plt.xticks(rotation=45)
    
    if overlap is None:  
        sns.lineplot(data[0], x=x, y=y, ci=None, ax=ax)
        if x == 'id':
            for xposition in range(365, 2190, 365):
                # 垂直線を描く
                ax.axvline(xposition, color='orange', lw=2)
                ax.set_xticks(np.arange(0, 2190, 365), [0, 365, 730, 1095, 1460, 1825])
    
    # 重ねてプロット
    else:
        
        for i in range(len(data)):  
            sns.lineplot(data[0], x=x, y=y, ci=None, label=y)
            
            for j in range(len(overlap)):
                sns.lineplot(data=data[i], x=x,
                             y=overlap[j], ci=None, label=overlap[j])
                if x == 'id':
                    for xposition in range(365, 2190, 365):
                        ax.axvline(xposition, color='orange', lw=2)
                        ax.set_xticks(np.arange(0, 2190, 365), [0, 365, 730, 1095, 1460, 1825])


draw_features([train], 'id', 'pressure')


draw_features([train], 'id', 'temparature', overlap=['mintemp', 'maxtemp'])


draw_features([train], 'day_of_year', 'diff_temp')


draw_features([train], 'id', 'diff_maxmin')


draw_features([train], 'id', 'dewpoint')


draw_features([train], 'id', 'humidity')


draw_features([train], 'id', 'cloud')


draw_features([train], 'id', 'sunshine')


draw_features([train], 'id', 'winddirection')


draw_features([train], 'id', 'windspeed')


draw_features([train], 'id', 'rainfall')


def draw_targets(data, x=None, y=None):
    
    sns.set(font_scale=2.0)
    fig, ax = plt.subplots(6, 1, figsize=(18,30))
    plt.xticks(rotation=45)
    plt.draw()
    for i in range(0, 6): 
        sns.lineplot(data[i], x=x, y=y, ci=None, ax=ax[i])

    plt.show()    


draw_targets([df_1, df_2, df_3, df_4, df_5, df_6], 'id', 'rainfall')


# the mean of 6 years 'rainfall'

fig, ax = plt.subplots(figsize=(18, 6))

rainfall_mean = train.groupby(['day_of_year'])['rainfall'].agg('mean').to_frame()

rainfall_mean = rainfall_mean.reset_index()
rainfall_mean.columns = ['day_of_year', 'rainfall']

sns.lineplot(data=rainfall_mean, x='day_of_year', y='rainfall', ci=None, ax=ax)
ax.set_title('mean value for 6 years', fontsize=12)

plt.tight_layout()
plt.show()


sns.set(font_scale=0.6)
plt.figure(figsize=(16,10))
sns.heatmap(train[train.columns[1:]].corr(), linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='YlGn'
);


fig, ax = plt.subplots(nrows=5, ncols=6,figsize=(12, 20))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(2, 7):
        col = train_df.columns[j]
        
        sns.histplot(
            data=train_df.iloc[start:end, :],
            x=col,
            ax=ax[j-2][i]
        )
        
        ax[j-2][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part train data> pressure, maxtemp, temparature, mintemp, dewpoint', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=2,figsize=(8, 12))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(2, 7):
        col = train_df.columns[j]
        
        sns.histplot(
            data=train_df.iloc[start:end, :],
            x=col,
            ax=ax[j-2][i]
        )
        
        ax[j-2][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part test data> pressure, maxtemp, temparature, mintemp, dewpoint', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=6,figsize=(12, 20))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(7, 12):
        col = train_df.columns[j]
        
        sns.histplot(
            data=train_df.iloc[start:end, :],
            x=col,
            ax=ax[j-7][i]
        )
        
        ax[j-7][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part train data> humidity, cloud, sunshine, winddirection, windspeed', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=5, ncols=2,figsize=(8, 12))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(7, 12):
        col = train_df.columns[j]
        
        sns.histplot(
            data=train_df.iloc[start:end, :],
            x=col,
            ax=ax[j-7][i]
        )
        
        ax[j-7][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part test data> humidity, cloud, sunshine, winddirection, windspeed', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=6,figsize=(12, 10))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 2190, 365)):
    start = part
    end = part + 365
    
    for j in range(13, 15):
        col = train.columns[j]
        
        sns.histplot(
            data=train.iloc[start:end, :],
            x=col,
            ax=ax[j-13][i]
        )
        
        ax[j-13][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part train data> diff_temp, diff_maxmin', y=1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(8, 5))
sns.set(font_scale=1.0)
for i, part in enumerate(range(0, 730, 365)):
    start = part
    end = part + 365
    
    for j in range(13, 15):
        col = train.columns[j]
        
        sns.histplot(
            data=train.iloc[start:end, :],
            x=col,
            ax=ax[j-13][i]
        )
        
        ax[j-13][i].set_title(f'part : {part} ~ {end}')

plt.suptitle('<part test data> diff_temp, diff_maxmin', y=1.0)
plt.tight_layout()
plt.show()


# missing value: winddirection  

test[test.winddirection.isnull()]


# interpolate to fill missing value

test = test.interpolate(method='linear')
test.iloc[516:519]



for col in train.columns[2:12]:
    count = train[col].nunique()
    print('train data:', col, count)

for col in test.columns[2:12]:
    count = test[col].nunique()
    print('test data:', col, count)


# check the angle of winddirection
print(train.winddirection.value_counts())


train[train.winddirection.eq(250.3)].index
train.iloc[1969:1972]['winddirection']


# winddirection

train[train.winddirection.eq(250.3)] = 250
train.iloc[1969:1972]['winddirection']


print(train.shape, test.shape)


cols = ['day_of_year', 
        'pressure', 
        'cloud', 
        'sunshine',
        'winddirection', 
        'windspeed',
        'diff_temp',
        'diff_maxmin',
        ]

X = train[cols]

results = []

for num in [730, 1095, 1460, 1825]:

    X_train = X.iloc[0:num]
    X_valid = X.iloc[num:num+365]
    y_train = train.iloc[0:num, 12]
    y_valid = train.iloc[num:num+365, 12]

    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    
    sc = StandardScaler()
    sc.fit(X_train)
    
    # standardization
    X_std = sc.transform(X_train)
    X_valid_std = sc.transform(X_valid)

    lr = LogisticRegression(C=0.9)
    lr.fit(X_std, y_train)
    
    y_pred = lr.predict_proba(X_valid_std)[:,1]
    
    fpr, tpr, thresholds = roc_curve(y_valid, np.round(y_pred, 1))
    acc = roc_auc_score(y_valid, np.round(y_pred, 1))

    results.append([lr, y_pred, acc])
    
    print(acc)
    plt.plot(fpr, tpr, marker='o', color='blue', markersize=1.0)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.grid()
    plt.show()

test_sc = sc.transform(test[cols])

result_df = pd.DataFrame(results)
result_df.columns = ['model', 'lm_pred', 'lm_acc']


model = result_df['model'][0]
pred = model.predict_proba(test_sc)[:,1]


submission = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(pred)),  
    "rainfall": np.squeeze(pred)  
})
submission.head()


submission.to_csv('LogisticRegression_1.csv', index=False)


from sklearn.neural_network import MLPClassifier

nnet_mlpc = MLPClassifier(random_state=42,
                         hidden_layer_sizes=(100,),
                         activation='relu',
                         alpha=0.5,
                         solver='lbfgs',
                         max_iter=5000,
                         ).fit(X_std, y_train)

print(nnet_mlpc.score(X_std, y_train))
print(nnet_mlpc.score(X_valid_std, y_valid))


mlpc_pred = nnet_mlpc.predict_proba(test_sc)


submission_3 = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(pred)),  
    "rainfall": np.squeeze(mlpc_pred[:,1])  
})
submission_3.head()


submission_3.to_csv('MLPClassifier_1.csv', index=False)


# add new features  year, month, day
# year :temporary setting　but excluding leap years

day_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

train['year'] = 0
train['month'] = 0
train['day'] = 0


for i, split_end in enumerate(range(365, 2191, 365)):
    # year
    split_start = split_end - 365
    
    if i < 3:
        temp_plus = i + 2013     # i=0~2 -> 2013,2014,2015
    else:
        temp_plus = i + 2014     # i=3~5 -> 2017, 2018, 2019
        
    train.loc[split_start:split_end-1, 'year'] = temp_plus
    
    # month
    mon_start = split_end - 365
    
    for mon in np.arange(1, 13):
        
        train.loc[mon_start:mon_start + day_of_month[mon-1], 'month'] = mon
        mon_start += day_of_month[mon-1]

        # day
        day_start = split_end - 365
        
        for day in day_of_month:
            for j in np.arange(1, day + 1):
                train.loc[day_start:day_start + 1, 'day'] = j
                day_start += 1



# add new features  year, month, day
# year :temporary setting　but excluding leap years

day_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

test['year'] = 0
test['month'] = 0
test['day'] = 0


for i, split_end in enumerate(range(365, 731, 365)):
    # year
    split_start = split_end - 365

    temp_plus = i + 2021     # i=0~1 -> 2021,2022
    
    test.loc[split_start:split_end-1, 'year'] = temp_plus
    
    # month
    mon_start = split_end - 365
    
    for mon in np.arange(1, 13):
        
        test.loc[mon_start:mon_start + day_of_month[mon-1], 'month'] = mon
        mon_start += day_of_month[mon-1]

        # day
        day_start = split_end - 365
        
        for day in day_of_month:
            for j in np.arange(1, day + 1):
                test.loc[day_start:day_start + 1, 'day'] = j
                day_start += 1



# new features 
# 'lag_pressure', 'lag_temp', 'lag_dewpoint', 'lag_humidity'

train['lag_pressure'] = train['pressure'].shift(1)
day_1 = train.groupby('day_of_year')['pressure'].mean().iloc[0]
train['lag_pressure'].fillna(day_1, inplace=True)

test['lag_pressure'] = test['pressure'].shift(1)
day_1 = test.groupby('day_of_year')['pressure'].mean().iloc[0]
test['lag_pressure'].fillna(day_1, inplace=True)


train['lag_temp'] = train['temparature'].shift(1)
day_1 = train.groupby('day_of_year')['temparature'].mean().iloc[0]
train['lag_temp'].fillna(day_1, inplace=True)

test['lag_temp'] = test['temparature'].shift(1)
day_1 = test.groupby('day_of_year')['temparature'].mean().iloc[0]
test['lag_temp'].fillna(day_1, inplace=True)

train['lag_dewpoint'] = train['dewpoint'].shift(1)
day_1 = train.groupby('day_of_year')['dewpoint'].mean().iloc[0]
train['lag_dewpoint'].fillna(day_1, inplace=True)

test['lag_dewpoint'] = test['dewpoint'].shift(1)
day_1 = test.groupby('day_of_year')['dewpoint'].mean().iloc[0]
test['lag_dewpoint'].fillna(day_1, inplace=True)

train['lag_humidity'] = train['humidity'].shift(1)
day_1 = train.groupby('day_of_year')['humidity'].mean().iloc[0]
train['lag_humidity'].fillna(day_1, inplace=True)

test['lag_humidity'] = test['humidity'].shift(1)
day_1 = test.groupby('day_of_year')['humidity'].mean().iloc[0]
test['lag_humidity'].fillna(day_1, inplace=True)


# new features 
# 'lag_cloud', 'lag_sun', 'lag_wind_d', 'lag_wind_s'

train['lag_cloud'] = train['cloud'].shift(1)
day_1 = train.groupby('day_of_year')['cloud'].mean().iloc[0]
train['lag_cloud'].fillna(day_1, inplace=True)

test['lag_cloud'] = test['cloud'].shift(1)
day_1 = test.groupby('day_of_year')['cloud'].mean().iloc[0]
test['lag_cloud'].fillna(day_1, inplace=True)


train['lag_sun'] = train['sunshine'].shift(1)
day_1 = train.groupby('day_of_year')['sunshine'].mean().iloc[0]
train['lag_sun'].fillna(day_1, inplace=True)

test['lag_sun'] = test['sunshine'].shift(1)
day_1 = test.groupby('day_of_year')['sunshine'].mean().iloc[0]
test['lag_sun'].fillna(day_1, inplace=True)


train['lag_wind_d'] = train['winddirection'].shift(1)
day_1 = train.groupby('day_of_year')['winddirection'].mean().iloc[0]
train['lag_wind_d'].fillna(day_1, inplace=True)

test['lag_wind_d'] = test['winddirection'].shift(1)
day_1 = test.groupby('day_of_year')['winddirection'].mean().iloc[0]
test['lag_wind_d'].fillna(day_1, inplace=True)


train['lag_wind_s'] = train['windspeed'].shift(1)
day_1 = train.groupby('day_of_year')['windspeed'].mean().iloc[0]
train['lag_wind_s'].fillna(day_1, inplace=True)

test['lag_wind_s'] = test['windspeed'].shift(1)
day_1 = test.groupby('day_of_year')['windspeed'].mean().iloc[0]
test['lag_wind_s'].fillna(day_1, inplace=True)


# new features
# 'diff_humidity', 'diff_cloud'

train['diff_humidity'] = train['humidity'].diff()
train['diff_humidity'].fillna(0, inplace=True)
test['diff_humidity'] = test['humidity'].diff()
test['diff_humidity'].fillna(0, inplace=True)

train['diff_cloud'] = train['cloud'].diff()
train['diff_cloud'].fillna(0, inplace=True)
test['diff_cloud'] = test['cloud'].diff()
test['diff_cloud'].fillna(0, inplace=True)


# new features
# 'diff_pressure' 'diff_temp'

train['diff_pressure'] = train['pressure'].diff()
train['diff_pressure'].fillna(0, inplace=True)
test['diff_pressure'] = test['pressure'].diff()
test['diff_pressure'].fillna(0, inplace=True)

train['diff_temp'] = train['diff_temp'].diff()
train['diff_temp'].fillna(0, inplace=True)
test['diff_temp'] = test['diff_temp'].diff()
test['diff_temp'].fillna(0, inplace=True)


# Train data has 29 features and 2190 records in all.
# Test data has 28 features and 730 records in all.

print(train.shape, test.shape)


# except 'rainfall'

cols = ['id', 'day_of_year', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'diff_temp', 'diff_maxmin', 'year', 'month',
       'day', 'lag_pressure', 'lag_temp', 'lag_dewpoint', 'lag_humidity',
       'loa_cloud', 'lag_sun', 'lag_wind_d', 'lag_wind_s', 'diff_humidity',
       'diff_cloud', 'diff_pressure']


X = train[cols]
y = train['rainfall']
feat_labels = X.columns

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print('%2d) %-*s %f' % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.figure(figsize=(10, 4))
plt.title('Feature importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=45)
plt.tick_params(labelsize=8)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()



X = train[cols]

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(X)

nums = np.arange(29)

var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(4,2), dpi=150)
plt.grid()
plt.plot(nums, var_ratio, marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')
plt.show()


# t-SNE: The effect of various perplexity values on the shape

perplexities = [5, 30, 50, 100]

cmp = plt.get_cmap('Set2')

fig, ax = plt.subplots(1, 4, figsize=(20, 4))

for i, perplexity in enumerate(perplexities):
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne_train = tsne.fit_transform(scaled_df)
    
    for j in range(2):
        select_rain = y == j
        plt_latent = X_tsne_train[select_rain, :]
        ax[i].scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(j), marker=f"${j}$")
        ax[i].xaxis.set_major_formatter(NullFormatter())
        ax[i].yaxis.set_major_formatter(NullFormatter())
        ax[i].set_title("Perplexity=%d" % perplexity)

plt.legend()
plt.suptitle('t-SNE')
plt.show()


# Check for duplicates   all data

X = train[cols]
y = train['rainfall']
test_X = test[cols]

# Check for duplicates in the datasets
print(f'Train data duplicates: {X.duplicated().sum()}')
print(f'Test data duplicates: {test_X.duplicated().sum()}')


# normalization
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X)
X_train_sc = scaler.transform(X)
X_test_sc = scaler.transform(test_X)

X_train, X_val, y_train, y_val = train_test_split(X_train_sc, 
                                                  y, 
                                                  test_size=0.2, 
                                                  random_state=42)


# Demonsinality reduction

pca = PCA(n_components=18)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

X_test_pca = pca.transform(X_test_sc)

print(f'Number of components retained: {pca.n_components_}')


# data Visualization :train data

df_pca_train = pd.DataFrame(X_train_pca) 
print(df_pca_train.shape)
df_pca_train.head()


# data Visualization :test data

df_pca_test = pd.DataFrame(X_test_pca) 
print(df_pca_test.shape)
df_pca_test.head()


# Percentage of variance explained by each of the selected components.

explained_variance = pca.explained_variance_ratio_

print(f'Percentage of variance explained: {explained_variance}')
print(f'Total variance explained: {sum(explained_variance)}')


# model performance metrics

# Initialize the Logistic Regression model
lr = LogisticRegression(max_iter=1000, C=0.9, random_state=42)

# Train the model on the PCA-transformed training data
lr.fit(X_train_pca, y_train)

# Predict on the PCA-transformed test set
y_pred_lr_pca = lr.predict(X_val_pca)

# Evaluate the model
print('-'*10, 'LogisticRegression:', '-'*10)
print('Accuracy:', accuracy_score(y_val, y_pred_lr_pca))
print('Precision:', precision_score(y_val, y_pred_lr_pca))
print('Recall:', recall_score(y_val, y_pred_lr_pca))
print('F1 Score:', f1_score(y_val, y_pred_lr_pca))
print(classification_report(y_val, y_pred_lr_pca))
print(confusion_matrix(y_val, y_pred_lr_pca))


sns.heatmap(confusion_matrix(y_val, y_pred_lr_pca), 
            linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='YlGn')
plt.xlabel('y_pred')
plt.ylabel('y_test');


# model performance metrics

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

y_pred_rf_pca = rf.predict(X_val_pca)

print('-'*10, 'RandomForestClassifier', '-'*10)
print('Accuracy:', accuracy_score(y_val, y_pred_rf_pca))
print('Precision:', precision_score(y_val, y_pred_rf_pca))
print('Recall:', recall_score(y_val, y_pred_rf_pca))
print('F1 Score:', f1_score(y_val, y_pred_rf_pca))
print(classification_report(y_val, y_pred_rf_pca))
print(confusion_matrix(y_val, y_pred_rf_pca))


sns.heatmap(confusion_matrix(y_val, y_pred_rf_pca), 
            linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='YlGn')
plt.xlabel('y_pred')
plt.ylabel('y_test');


# model performance metrics

dt = DecisionTreeClassifier()
dt.fit(X_train_pca, y_train)

y_pred_dt_pca = dt.predict(X_val_pca)

print('-'*10, 'DecisionTreeClassifier', '-'*10)
print('Accuracy:', accuracy_score(y_val, y_pred_dt_pca))
print('Precision:', precision_score(y_val, y_pred_dt_pca))
print('Recall:', recall_score(y_val, y_pred_dt_pca))
print('F1 Score:', f1_score(y_val, y_pred_dt_pca))
print(classification_report(y_val, y_pred_dt_pca))
print(confusion_matrix(y_val, y_pred_dt_pca))


sns.heatmap(confusion_matrix(y_val, y_pred_dt_pca), 
            linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='YlGn')
plt.xlabel('y_pred')
plt.ylabel('y_test');


# model performance metrics

nb = GaussianNB()
nb.fit(X_train_pca, y_train)

y_pred_nb_pca = nb.predict(X_val_pca)

print('-'*10, 'Gaussian Naive Bayes', '-'*10)
print('Accuracy:', accuracy_score(y_val, y_pred_nb_pca))
print('Precision:', precision_score(y_val, y_pred_nb_pca))
print('Recall:', recall_score(y_val, y_pred_nb_pca))
print('F1 Score:', f1_score(y_val, y_pred_nb_pca))
print(classification_report(y_val, y_pred_nb_pca))
print(confusion_matrix(y_val, y_pred_nb_pca))


sns.heatmap(confusion_matrix(y_val, y_pred_nb_pca), 
            linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='YlGn')
plt.xlabel('y_pred')
plt.ylabel('y_test');


# submit

pred_lr = lr.predict_proba(X_test_pca)[:,1]

submission_lr = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(pred_lr)),  
    "rainfall": np.squeeze(pred_lr)  
})
submission_lr.head()


submission_lr.to_csv('LogisticRegression_5.csv', index=False)


pred_rf = rf.predict_proba(X_test_pca)[:,1]

preds = []
preds.append(pred_lr)
preds.append(pred_rf)

preds_mean = np.mean(np.array(preds), axis=0)


submission_ans = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(preds_mean)),  
    "rainfall": np.squeeze(preds_mean)  
})
submission_ans.head()


submission_ans.to_csv('RandomForest_LogisticRegression_ans_1.csv', index=False)


# LightGBM

from sklearn.metrics import log_loss

cols = ['id', 'day_of_year', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 
       'windspeed','year', 'month', 'day', 'lag_pressure',
       'lag_temp', 'lag_dewpoint', 'lag_humidity', 'lag_cloud', 'lag_sun',
       'lag_wind_d', 'lag_wind_s', 'diff_temp', 'diff_maxmin', 'diff_humidity',
       'diff_cloud', 'diff_pressure']

X = train[cols]
y = train['rainfall']
test_X = test[cols]

# normalization
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X)
X_train_sc = scaler.transform(X)
X_test_sc = scaler.transform(test_X)

lgb_params = {
    'objective': 'binary',
    'n_estimator': 1000,
    'learning_rate':0.01,
    'num_leaves': 10,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'feature_fraction': 1.0,
    'max_depth': 5,
    'min_child_samples': 10,
    'min_sum_hessian_in_leaf': 1,
    'eval_metric': 'logloss',
    'lambda_l1': 0.01,
    'lambda_l2': 0,
    'random_seed': 42,
    'verbosity': -1
}

models_lgb = []
acc_lgb = []
oof_lgb = np.zeros(len(X))


for i, num in enumerate([730, 1095, 1460, 1825]):
    X_train = X_train_sc[0:num]
    X_valid = X_train_sc[num:num+365]
    y_train = train.iloc[0:num, 12]
    y_valid = train.iloc[num:num+365, 12]
    
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    print('-'*5, i+1, ' times', '-'*5)
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)


    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=500,
                                      verbose=True), 
                   lgb.log_evaluation(100)],
    )
    
    va_pred = model_lgb.predict(X_valid, 
                                num_iteration=model_lgb.best_iteration)

    models_lgb.append(model_lgb)
    
    score = log_loss(y_valid, va_pred)
    fpr, tpr, thresholds = roc_curve(y_valid, np.round(va_pred, 1))
    tmp_acc = roc_auc_score(y_valid, np.round(va_pred, 1))

    print(f'log_loss: {score:.4f}')
    print(f'acc: {tmp_acc:.4f}')

    plt.plot(fpr, tpr, marker='o', color='blue', markersize=1.0)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title('LightGBM')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.grid()
    plt.show()
    
    models_lgb.append(model_lgb)
    acc_lgb.append(tmp_acc)

    if i == 0:
        s_index = 0
        e_index = train.iloc[365, 0]
        oof_lgb[s_index:e_index] = va_pred
    elif i < 3:
        s_index = train.iloc[num, 0]
        e_index = train.iloc[num+365, 0]
        oof_lgb[s_index:e_index] = va_pred
    else:
        s_index = train.iloc[num, 0]
        oof_lgb[s_index:] = va_pred


preds_lgb = []

for model in models_lgb:
    pred = model.predict(X_test_sc)
    preds_lgb.append(pred)

preds_lgb_array = np.array(preds_lgb)
preds_lgb_mean = np.mean(preds_lgb_array, axis=0)

submission = pd.DataFrame({
    "id": np.arange(2190, 2190 + len(preds_lgb_mean)),  
    "rainfall": np.squeeze(preds_lgb_mean)  
})
print(submission.shape)
submission.head()


submission.to_csv('LightGBM_1.csv', index=False)





# nunique:35 (angle: 0°~360°) north: 0 east: 90 south: 180 west:270
# Wind blowing from east to west and from west to east 

wind_d_df = pd.DataFrame(train.groupby('winddirection').rainfall.count())
wind_d_df.columns = ['rainfall']
wind_d_df.rainfall.astype('int')

wind_d_df


def mapping_winddirection(x):
    if (x >= 40) & (x <= 130):
        direction = 3
    elif (x >= 140) & (x <= 230):
        direction = 2 
    elif (x >= 240) & (x <= 330):
        direction = 1 
    else:
        direction = 0
        
    return direction


def mapping_windspeed(x):
    if x < 10:
        level = 0
    elif (x >= 10) & (x < 15):
        level = 1 
    elif (x >= 15) & (x < 20):
        level = 2
    elif (x >= 20) & (x < 30):
        level = 3
    elif (x >= 30):
        level = 4
        
    return level
        


# add features : 'winddirection_from', 'windspeed_level', 'diff_pressure', 'diff_wind_d'

train['winddirection_from'] = train['winddirection'].map(mapping_winddirection)
test['winddirection_from'] = test['winddirection'].map(mapping_winddirection)

train['windspeed_level'] = train['windspeed'].map(mapping_windspeed)
test['windspeed_level'] = test['windspeed'].map(mapping_windspeed)

train['diff_pressure'] = np.abs(train['pressure'] - train['lag_pressure'])
test['diff_pressure'] = np.abs(test['pressure'] - test['lag_pressure'])

train['diff_wind_d'] = np.abs(train['winddirection'] - train['lag_wind_d'])
test['diff_wind_d'] = np.abs(test['winddirection'] - test['lag_wind_d'])


train_pre_mean = train.groupby(['month', 'day']).pressure.mean()
train_pre_mean.columns = ['pressure_day_mean']

train['pressure_day_mean'] = pd.concat([train_pre_mean,
                                        train_pre_mean,
                                        train_pre_mean,
                                        train_pre_mean,
                                        train_pre_mean,
                                        train_pre_mean]).values


test_pre_mean = test.groupby(['month', 'day']).pressure.mean()
test_pre_mean.columns = ['pressure_day_mean']

test['pressure_day_mean'] = pd.concat([test_pre_mean, test_pre_mean]).values


train_temp_mean = train.groupby(['month', 'day']).temparature.mean()
train_temp_mean.columns = ['temparature_day_mean']

train['temparature_day_mean'] = pd.concat([train_temp_mean,
                                        train_temp_mean,
                                        train_temp_mean,
                                        train_temp_mean,
                                        train_temp_mean,
                                        train_temp_mean]).values


test_temp_mean = test.groupby(['month', 'day']).temparature.mean()
test_temp_mean.columns = ['temp_day_mean']

test['temparature_day_mean'] = pd.concat([test_temp_mean, test_temp_mean]).values



train_dew_mean = train.groupby(['month', 'day']).dewpoint.mean()
train_dew_mean.columns = ['dewpoint_day_mean']

train['dewpoint_day_mean'] = pd.concat([train_dew_mean,
                                        train_dew_mean,
                                        train_dew_mean,
                                        train_dew_mean,
                                        train_dew_mean,
                                        train_dew_mean]).values


test_dew_mean = test.groupby(['month', 'day']).dewpoint.mean()
test_dew_mean.columns = ['dewpoint_day_mean']

test['dewpoint_day_mean'] = pd.concat([test_dew_mean, test_dew_mean]).values


train_hum_mean = train.groupby(['month', 'day']).humidity.mean()
train_hum_mean.columns = ['humidity_day_mean']

train['humidity_day_mean'] = pd.concat([train_hum_mean,
                                        train_hum_mean,
                                        train_hum_mean,
                                        train_hum_mean,
                                        train_hum_mean,
                                        train_hum_mean]).values


test_hum_mean = test.groupby(['month', 'day']).humidity.mean()
test_hum_mean.columns = ['humidity_day_mean']

test['humidity_day_mean'] = pd.concat([test_hum_mean, test_hum_mean]).values


train_cloud_mean = train.groupby(['month', 'day']).cloud.mean()
train_cloud_mean.columns = ['cloud_day_mean']

train['cloud_day_mean'] = pd.concat([train_cloud_mean,
                                     train_cloud_mean,
                                     train_cloud_mean,
                                     train_cloud_mean,
                                     train_cloud_mean,
                                     train_cloud_mean]).values


test_cloud_mean = test.groupby(['month', 'day']).cloud.mean()
test_cloud_mean.columns = ['cloud_day_mean']

test['cloud_day_mean'] = pd.concat([test_cloud_mean, test_cloud_mean]).values


train_sun_mean = train.groupby(['month', 'day']).sunshine.mean()
train_sun_mean.columns = ['sunshine_day_mean']

train['sunshine_day_mean'] = pd.concat([train_sun_mean,
                                        train_sun_mean,
                                        train_sun_mean,
                                        train_sun_mean,
                                        train_sun_mean,
                                        train_sun_mean]).values


test_sun_mean = test.groupby(['month', 'day']).sunshine.mean()
test_sun_mean.columns = ['sunshine_day_mean']

test['sunshine_day_mean'] = pd.concat([test_sun_mean, test_sun_mean]).values


train.shape, test.shape


cols = ['id', 'day_of_year', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'year', 'month', 'day', 'lag_pressure',
       'lag_temp', 'lag_dewpoint', 'lag_humidity', 'lag_cloud', 'lag_sun',
       'lag_wind_d', 'lag_wind_s', 'diff_temp', 'diff_maxmin', 'diff_humidity',
       'diff_cloud', 'windspeed_level', 'winddirection_from', 'diff_wind_d',
       'diff_pressure', 'pressure_day_mean', 'temparature_day_mean',
       'dewpoint_day_mean', 'humidity_day_mean', 'cloud_day_mean',
       'sunshine_day_mean', 'diff_humidity_2', 'diff_cloud_2']


X = train[cols]
y = train['rainfall']
test_X = test[cols]

# normalization
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X)
X_train_sc = scaler.transform(X)
X_test_sc = scaler.transform(test_X)

pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)


sequence_length = 365
output_sequence = 30

X = train[cols][:len(train)-sequence_length]

y = train['rainfall'].values
y = y.astype('float32')
y_train = np.reshape(y, (-1, 1))

data_all = pd.concat([train[cols], test[cols]], axis=0)
test_X = data_all[cols][len(train)-sequence_length:]


# normalization
scaler_X = MinMaxScaler(feature_range=(0, 1))

X_train_sc = scaler_X.fit_transform(X)
X_test_sc = scaler_X.transform(test_X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_sc = scaler_y.fit_transform(y_train)

print(X_train_sc.shape, X_test_sc.shape, y_train_sc.shape)


def create_sequences(X, y, sequence_length=sequence_length, train=True):
    Xs, ys = [], []
    
    if train:
        for i in range(len(X) - sequence_length):
            
            Xs.append(X[i:sequence_length + i])
            ys.append(y[sequence_length + i])
    else:        
        for i in range(len(X) - sequence_length):
            Xs.append(X[i:sequence_length + i])

    return np.array(Xs), np.array(ys)


X_sequence, y_sequence = create_sequences(X_train_sc, y_train_sc, sequence_length=sequence_length)

test_X, _ = create_sequences(X_test_sc, None, sequence_length=sequence_length, train=False)

split = int(2/3 * len(X_sequence))
print(split)
X_train, X_valid = X_sequence[:split], X_sequence[split:]
y_train, y_valid = y_sequence[:split], y_sequence[split:]

print(X_sequence.shape, y_sequence.shape)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
print(test_X.shape)


inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))

x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64)(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train,
                    epochs=1000,
                    batch_size=32,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             mode='min',
                                                             patience=20),
                               keras.callbacks.ModelCheckpoint('jena_lstm.keras', 
                                                               save_best_only=True)],
                    verbose=1,)    


model_lstm = keras.models.load_model('jena_lstm.keras')

