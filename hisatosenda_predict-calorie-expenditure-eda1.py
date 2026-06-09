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


!pip install pytorch-tabnet


import os
import random

%matplotlib inline
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets
from torchinfo import summary
from pytorch_tabnet.tab_model import TabNetRegressor

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from glob import glob
import cv2
# from tqdm.auto import tqdm
from tqdm import tqdm
from PIL import Image
import pathlib

import warnings
warnings.filterwarnings('ignore')


def seed_torch(seed=24):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#    torch.backends.cudnn.deterministic = True   # PyTorchの畳み込み演算の再現性を確保
#    torch.use_deterministic_algorithms = True   # 「決定論的」アルゴリズムを使用



submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.head(3)


test_df.head(3)


submission_df.head(2)


train_df.isnull().sum()


test_df.isnull().sum()


print(train_df.shape, test_df.shape)


print(train_df.nunique(), '\n', test_df.nunique())


print(train_df.info())
print(test_df.info())


all_df = pd.concat([train_df, test_df])
all_df.Sex.unique()


all_df['Sex'] = all_df['Sex'].map({'male': 0, 'female': 1})


all_df['Sex'].value_counts()


all_df.iloc[:,1:].describe().T


train_df = all_df[:750000]
test_df = all_df.iloc[750000:, :-1]

del all_df


from mlxtend.plotting import heatmap

df = train_df.iloc[:, 1:]
cm = np.corrcoef(df.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns, 
             figsize=(6,6), cmap='BuGn')
plt.tight_layout()
plt.show()


# Male

train_male_df = train_df.loc[train_df.Sex.eq(0)].iloc[:, 2:]

sns.set_theme(style="ticks")
sns.pairplot(train_male_df)
plt.title('Scatter Matrix [Male]')
plt.tight_layout()
plt.show()


# Female

train_female_df = train_df.loc[train_df.Sex.eq(1)].iloc[:, 2:]

sns.set_theme(style="ticks")
sns.pairplot(train_female_df)
plt.title('Scatter Matrix [Female]')
plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# LinearRegression model plot

from sklearn.linear_model import LinearRegression

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

fig, ax = plt.subplots(2, 3, figsize=(12, 5))

for i in range(2):
    for j, col in enumerate(cols):
        X = train_df.loc[train_df.Sex.eq(i), [col]].values
        y = train_df.loc[train_df.Sex.eq(i), 'Calories'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        y_pred = lr.predict(X)
        
        ax[i][j].scatter(X, y, c='steelblue', edgecolor='white', s=70)  # s:size
        ax[i][j].plot(X, lr.predict(X), color='black', lw=2)            # lw:linewidth
        ax[i][j].set_xlabel(col)
        ax[i][j].set_ylabel('Calories')
        if i == 0:
            ax[i][j].set_title('Male')
        else:
            ax[i][j].set_title('Female')
    
plt.suptitle('Linear Regression Model')
plt.tight_layout()
plt.show()


# Residual plot: LinearRegression model

target ='Calories'
features = train_df.columns[train_df.columns != target]

X = train_df[features].values
y = train_df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

ax[0].scatter(y_test_pred, (y_test_pred - y_test), 
              c='limegreen', marker='s', edgecolor='white', label='Test data')
ax[0].set_ylabel('Residuals')

ax[1].scatter(y_train_pred, (y_train_pred - y_train), 
              c='steelblue', marker='o', edgecolor='white', label='Training data')

for ax in (ax[0], ax[1]):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left', fontsize=8)
    ax.hlines(y=0, 
              xmin=(x_min - 100),
              xmax=(x_max + 100), color='black', lw=2)

plt.suptitle('LinearRegression model residuals ')
plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# DecisionTreeRegressor model plot

from sklearn.tree import DecisionTreeRegressor

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

fig, ax = plt.subplots(2, 3, figsize=(12, 5))

for i in range(2):
    for j, col in enumerate(cols):
        X = train_df.loc[train_df.Sex.eq(i), [col]].values
        y = train_df.loc[train_df.Sex.eq(i), 'Calories'].values
        
        tree = DecisionTreeRegressor(max_depth=5)
        tree.fit(X, y)
        
        sort_idx = X.flatten().argsort()
        
        ax[i][j].scatter(X[sort_idx], y[sort_idx], 
                         c='steelblue', edgecolor='white', s=70)       
        ax[i][j].plot(X[sort_idx], 
                      tree.predict(X[sort_idx]), color='black', lw=2)  
        ax[i][j].set_xlabel(col)
        ax[i][j].set_ylabel('Calories')
        if i == 0:
            ax[i][j].set_title('Male')
        else:
            ax[i][j].set_title('Female')
    
plt.suptitle('DecisionTreeRegressor Model')
plt.tight_layout()
plt.show()


# Residual plot: DecisionTreeRegressor model

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

ax[0].scatter(y_test_pred, (y_test_pred - y_test), 
              c='limegreen', marker='s', edgecolor='white', label='Test data')
ax[0].set_ylabel('Residuals')

ax[1].scatter(y_train_pred, (y_train_pred - y_train), 
              c='steelblue', marker='o', edgecolor='white', label='Training data')

for ax in (ax[0], ax[1]):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left', fontsize=8)
    ax.hlines(y=0, 
              xmin=(x_min - 100),
              xmax=(x_max + 100), color='black', lw=2)

plt.suptitle('DecisionTreeRegressor model residuals')
plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# RandomForestRegressor model plot

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

fig, ax = plt.subplots(2, 3, figsize=(12, 5))

for i in range(2):
    for j, col in enumerate(cols):
        X = train_df.loc[train_df.Sex.eq(i), [col]].values
        y = train_df.loc[train_df.Sex.eq(i), 'Calories'].values
        
        forest = RandomForestRegressor(n_estimators=500,
                                       criterion='squared_error',
                                       random_state=1, n_jobs=-1)
        forest.fit(X, y)
        
        sort_idx = X.flatten().argsort()
        
        ax[i][j].scatter(X[sort_idx], y[sort_idx], 
                         c='steelblue', edgecolor='white', s=70)       
        ax[i][j].plot(X[sort_idx], 
                      forest.predict(X[sort_idx]), color='black', lw=2)  
        ax[i][j].set_xlabel(col)
        ax[i][j].set_ylabel('Calories')
        if i == 0:
            ax[i][j].set_title('Male')
        else:
            ax[i][j].set_title('Female')
    
plt.suptitle('RandomForestRegressor Model')
plt.tight_layout()
plt.show()


# Residual plot: RandomForestRegressor model

forest = RandomForestRegressor(n_estimators=500,
                               criterion='squared_error',
                               random_state=1, n_jobs=-1)

forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

ax[0].scatter(y_test_pred, (y_test_pred - y_test), 
              c='limegreen', marker='s', edgecolor='white', label='Test data')
ax[0].set_ylabel('Residuals')

ax[1].scatter(y_train_pred, (y_train_pred - y_train), 
              c='steelblue', marker='o', edgecolor='white', label='Training data')

for ax in (ax[0], ax[1]):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left', fontsize=8)
    ax.hlines(y=0, 
              xmin=(x_min - 100),
              xmax=(x_max + 100), color='black', lw=2)

plt.suptitle('RandomForestRegressor model residuals')
plt.tight_layout()
plt.show()


plt.clf()
plt.close()


# Difference between y_pred and y_test

pred_df = pd.DataFrame({'id': X_test[:,0],'y_pred': y_test_pred, 
                        'Calories': y_test, 'diff': y_test_pred - y_test})

pred_df.head(3)


# Predicted values are very small
display(pred_df.loc[pred_df['diff'] < -100])

# Predicted values are very big
display(pred_df.loc[pred_df['diff'] > 100])


# 
print(pred_df.loc[pred_df['diff'] < -100, 'id'])
print(pred_df.loc[pred_df['diff'] > 100, 'id'])


# Residuals are less than minus 100

resid_id = pred_df.loc[pred_df['diff'] < -100, 'id']

df = pd.DataFrame(X_test, columns=['id', 'Sex', 'Age', 'Height', 'Weight', 
                                   'Duration', 'Heart_Rate', 'Body_Temp'])

resid_minus100_df = df.loc[df['id'].isin(resid_id)]
resid_minus100_df['Calories'] = pred_df.loc[pred_df['id'].isin(resid_id), 'Calories'].values
resid_minus100_df['pred'] = pred_df.loc[pred_df['id'].isin(resid_id), 'y_pred'].values
resid_minus100_df['diff'] = pred_df.loc[pred_df['id'].isin(resid_id), 'diff'].values
resid_minus100_df


# Male
resid_male_id = pred_df.loc[(pred_df['diff'] < -100), 'id']
resid_minus100_df.loc[resid_minus100_df.id.isin(resid_male_id) & resid_minus100_df.Sex.eq(0)].iloc[:,2:9].describe().T


# Compare residuals less than minus 100 and values of training data (Male)

train_male_df.describe().T


# Female
resid_female_id = pred_df.loc[(pred_df['diff'] < -100), 'id']
resid_minus100_df.loc[resid_minus100_df.id.isin(resid_female_id) & resid_minus100_df.Sex.eq(1)].iloc[:,2:9].describe().T


# Compare residuals less than minus 100 and values of training data (Female)

train_female_df.describe().T


resid_minus100_df


# Residuals are more than 100

resid_id = pred_df.loc[pred_df['diff'] > 100, 'id']

df = pd.DataFrame(X_test, columns=['id', 'Sex', 'Age', 'Height', 'Weight', 
                                   'Duration', 'Heart_Rate', 'Body_Temp'])

resid_plus100_df = df.loc[df['id'].isin(resid_id)]
resid_plus100_df['Calories'] = pred_df.loc[pred_df['id'].isin(resid_id), 'Calories'].values
resid_plus100_df['pred'] = pred_df.loc[pred_df['id'].isin(resid_id), 'y_pred'].values
resid_plus100_df['diff'] = pred_df.loc[pred_df['id'].isin(resid_id), 'diff'].values

resid_plus100_df


# Compare residuals more than 100 and values of training data (Male)

train_male_df.describe().T


all_df = pd.concat([train_df, test_df])


# Bucket age

boundaries_age = torch.tensor([28, 40, 52])
v = torch.tensor(all_df['Age'].values)

# 20~27, 28~39, 40~51, 52~79
all_df['Age_Bucketed'] = torch.bucketize(v, boundaries_age, right=True)



# Bucket Duration

boundaries_du = torch.tensor([8, 15, 23])
v = torch.tensor(all_df['Duration'].values)

# 1~7, 8~14, 14~22, 23~30
all_df['Duration_Bucketed'] = torch.bucketize(v, boundaries_du, right=True)



# BMI: Body Mass Indes  
# healty values: 18.5~24.9

all_df['BMI'] = np.round(all_df['Weight'] / ((all_df['Height']/100)**2), 2)


# Male

all_df.loc[all_df.Sex.eq(0), 'BMI'].describe().T


# Female

all_df.loc[all_df.Sex.eq(1), 'BMI'].describe().T


# Bucket BMI　by gender

all_df['BMI_Bucketed'] = 0

# Male
boundaries_bmi = torch.tensor([24.67, 25.43, 26.19])
v = torch.tensor(all_df['BMI'].values)

# 14.30~, 24.67~, 25.43~, 26.19~
all_df['BMI_0'] = torch.bucketize(v, boundaries_bmi, right=True)
all_df.loc[all_df.Sex.eq(0), 'BMI_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'BMI_0']


# Female
boundaries_bmi = torch.tensor([22.58, 23.33, 24.11])
v = torch.tensor(all_df['BMI'].values)

# 12.37~, 22.58~, 23.33~, 24.11~
all_df['BMI_1'] = torch.bucketize(v, boundaries_bmi, right=True)
all_df.loc[all_df.Sex.eq(1), 'BMI_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'BMI_1']

all_df = all_df.drop(['BMI_0', 'BMI_1'], axis=1)


# BMR: Basal metabolic rate
# Men: BMR = 88.362 + (13.397 x weight in kg) + (4.799 x height in cm) – (5.677 x age in years)
# Women: BMR = 447.593 + (9.247 x weight in kg) + (3.098 x height in cm) – (4.330 x age in years)

all_df['BMR'] = 0

all_df['bmr_weight_0'] = 13.397 * all_df['Weight']
all_df['bmr_height_0'] = 4.799 * all_df['Height']
all_df['bmr_age_0'] = 5.677 * all_df['Age']
all_df['bmr_0'] = np.round(all_df['bmr_weight_0'] + all_df['bmr_height_0'] - all_df['bmr_age_0'] + 88.362, 2)
all_df.loc[all_df['Sex'].eq(0), 'BMR'] = all_df.loc[all_df['Sex'].eq(0), 'bmr_0']

all_df['bmr_weight_1'] = 9.247 * all_df['Weight']
all_df['bmr_height_1'] = 3.098 * all_df['Height']
all_df['bmr_age_1'] = 4.33 * all_df['Age']
all_df['bmr_1'] = np.round(all_df['bmr_weight_1'] + all_df['bmr_height_1'] - all_df['bmr_age_1'] + 447.593, 2)
all_df.loc[all_df['Sex'].eq(1), 'BMR'] = all_df.loc[all_df['Sex'].eq(1), 'bmr_1']


all_df = all_df.drop(['bmr_weight_0','bmr_height_0', 'bmr_age_0', 'bmr_0', 
                      'bmr_weight_1','bmr_height_1', 'bmr_age_1', 'bmr_1'], axis=1)
all_df.head(2)


# Male
all_df.loc[all_df['Sex'].eq(0), 'BMR'].describe().T


# Female
all_df.loc[all_df['Sex'].eq(1), 'BMR'].describe().T


# Bucket BMR by gender

all_df['BMR_Bucketed'] = 0

# Male
boundaries_bmr = torch.tensor([1782, 1899, 2019])
v = torch.tensor(all_df['BMR'].values)

# 1032~, 1782~, 1899~, 2019~
all_df['BMR_0'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(0), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'BMR_0']

# Female
boundaries_bmr = torch.tensor([1296, 1368, 1443])
v = torch.tensor(all_df['BMR'].values)

# 940~, 1296~, 1368~, 1443~
all_df['BMR_1'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(1), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'BMR_1']

all_df = all_df.drop(['BMR_0', 'BMR_1'], axis=1)


# Healty_Weight = Height/100 * Height/100 * 22

all_df['Healty_Weight'] = np.round(((all_df['Height']/100) ** 2) * 22, 2)


# Male
all_df.loc[all_df['Sex'].eq(0), 'Healty_Weight'].describe().T


# Female
all_df.loc[all_df['Sex'].eq(1), 'Healty_Weight'].describe().T


# Bucket Healty Weight by gender

all_df['Healty_Weight_Bucketed'] = 0

# Male
boundaries_hel = torch.tensor([70.4, 74.4, 80.2])
v = torch.tensor(all_df['Healty_Weight'].values)

# 43.7~, 70.4~, 74.4~, 80.2~
all_df['hel_0'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'hel_0']


# Female
boundaries_hel = torch.tensor([55.6, 59.1, 64.3])
v = torch.tensor(all_df['Healty_Weight'].values)

# 34.9~, 55.6~, 59.1~, 64.3~
all_df['hel_1'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'hel_1']


all_df = all_df.drop(['hel_0', 'hel_1'], axis=1)


all_df.head(2)


# Difference 'between_Weight' and 'Healty_Weight'

all_df['Diff_Weight'] = np.round(all_df['Weight'] - all_df['Healty_Weight'], 2)


# 'Over_Weight': 'Diff Weight' <= 0 

all_df['Over_Weight'] = np.abs(all_df['Diff_Weight'])

all_df.loc[all_df['Diff_Weight'] <= 0, 'Over_Weight'] = 0


all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].describe().T


all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].describe().T


# Bucket Over Weight by gender

all_df['Over_Weight_Bucketed'] = 0

# Male
boundaries_ow = torch.tensor([8.93, 11.53, 14.32])
v = torch.tensor(all_df['Over_Weight'].values)

# 0~, 8.93~, 11.53~, 14.32~
all_df['ow_0'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'ow_0']


# Female
boundaries_ow = torch.tensor([1.55, 3.67, 5.77])
v = torch.tensor(all_df['Over_Weight'].values)

# 0~, 1.55~, 3.67~, 5.77~
all_df['ow_1'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'ow_1']


all_df = all_df.drop(['ow_0', 'ow_1'], axis=1)


# Plot each 'Age_Bucketed' (training data)

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

all_male_df = all_df.loc[(all_df.Sex.eq(0)) & ~(all_df.Calories.isnull())]
all_female_df = all_df.loc[(all_df.Sex.eq(1)) & ~(all_df.Calories.isnull())]

for i, col in enumerate(cols):
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
 
    sns.barplot(data=all_male_df, x=col, y='Calories', hue='Age_Bucketed', ax=ax[0])
    sns.barplot(data=all_female_df, x=col, y='Calories', hue='Age_Bucketed', ax=ax[1])
    ax[0].set_title(f'Male: {col}')
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[1].set_title(f'Female: {col}')
    ax[1].tick_params(axis='x', labelrotation=45)
    
    plt.legend(loc='best')
    plt.suptitle('Plot each Age Bucketed (Training data)')
    plt.tight_layout()
    plt.show()



# Plot each 'BMI_Bucketed' (training data)

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

# all_male_df = all_df.loc[(all_df.Sex.eq(0)) & ~(all_df.Calories.isnull())]
# all_female_df = all_df.loc[(all_df.Sex.eq(1)) & ~(all_df.Calories.isnull())]

for i, col in enumerate(cols):
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
 
    sns.barplot(data=all_male_df, x=col, y='Calories', hue='BMI_Bucketed', ax=ax[0])
    sns.barplot(data=all_female_df, x=col, y='Calories', hue='BMI_Bucketed', ax=ax[1])
    ax[0].set_title(f'Male: {col}')
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[1].set_title(f'Female: {col}')
    ax[1].tick_params(axis='x', labelrotation=45)

    plt.legend(loc='best')
    plt.suptitle('Plot each BMI Bucketed (Training data)')
    plt.tight_layout()
    plt.show()



plt.clf()
plt.close()


# Plot each 'Over_Weight_Bucketed' (training data)

cols = ['Duration', 'Heart_Rate', 'Body_Temp']

# all_male_df = all_df.loc[(all_df.Sex.eq(0)) & ~(all_df.Calories.isnull())]
# all_female_df = all_df.loc[(all_df.Sex.eq(1)) & ~(all_df.Calories.isnull())]

for i, col in enumerate(cols):
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
 
    sns.barplot(data=all_male_df, x=col, y='Calories', hue='Over_Weight_Bucketed', ax=ax[0])
    sns.barplot(data=all_female_df, x=col, y='Calories', hue='Over_Weight_Bucketed', ax=ax[1])
    ax[0].set_title(f'Male: {col}')
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[1].set_title(f'Female: {col}')
    ax[1].tick_params(axis='x', labelrotation=45)

    plt.legend(loc='best')
    plt.suptitle('Plot each Over_Weight_Bucketed (Training data)')
    plt.tight_layout()
    plt.show()


plt.clf()
plt.close()

del all_male_df, all_female_df


train_df = all_df[:750000]
test_df = all_df.iloc[750000:, :-1]

del all_df
print(train_df.shape, test_df.shape)


train_df.tail(2)


train_df.info()


# Group: 'Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed' (Training data)

df = train_df.groupby(['Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed'])['Calories'].agg('mean')

group_df = pd.DataFrame(df, columns=['Calories']).reset_index()
group_df.columns = ['Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed', 'Calories']
group_df.head(3)


# resid_minus100_df

resid_minus_id = resid_minus100_df['id']
resid_minus100_calories_df = train_df.loc[train_df.id.isin(resid_minus_id), ['id', 'Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed', 'Calories']]
resid_minus100_calories_df


# Differences between predicted calories and group's calories, 
# and between predicted calories and target valories.

resid_minus100_df['group_calories'] = 0

for i in range(resid_minus100_calories_df.shape[0]):
    num = resid_minus100_calories_df.iloc[i, 0]
    age = resid_minus100_calories_df.iloc[i, 1]
    duration = resid_minus100_calories_df.iloc[i, 2]
    bmi = resid_minus100_calories_df.iloc[i, 3]
    bmr = resid_minus100_calories_df.iloc[i, 4]
    
    print('Predicted calories: ', resid_minus100_df.loc[resid_minus100_df.id.eq(num), 'pred'].values, '/',
          "Group's calories  : ", group_df.loc[group_df.Age_Bucketed.eq(age) &  group_df.Duration_Bucketed.eq(duration) & 
                                  group_df.BMI_Bucketed.eq(bmi) & group_df.BMR_Bucketed.eq(bmr), 'Calories'].values)
    print('Predicted calories: ', resid_minus100_df.loc[resid_minus100_df.id.eq(num), 'pred'].values, '/',
          'Target calories   : ', train_df.loc[train_df.id.eq(num),  'Calories'].values, '\n')

    resid_minus100_df.loc[resid_minus100_df.id.eq(num), 'group_calories'] = group_df.loc[group_df.Age_Bucketed.eq(age) &  
                                                                            group_df.Duration_Bucketed.eq(duration) & 
                                                                            group_df.BMI_Bucketed.eq(bmi) & 
                                                                            group_df.BMR_Bucketed.eq(bmr), 'Calories'].values


display(resid_minus100_df)


# resid_plus100_df

resid_plus_id = resid_plus100_df['id']
resid_plus100_calories_df = train_df.loc[train_df.id.isin(resid_plus_id), ['id', 'Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed', 'Calories']]
resid_plus100_calories_df


# Differences between predicted calories and group's calories, 
# and between predicted calories and target valories.

resid_plus100_df['group_calories'] = 0

for i in range(resid_plus100_calories_df.shape[0]):
    num = resid_plus100_calories_df.iloc[i, 0]
    age = resid_plus100_calories_df.iloc[i, 1]
    duration = resid_plus100_calories_df.iloc[i, 2]
    bmi = resid_plus100_calories_df.iloc[i, 3]
    bmr = resid_plus100_calories_df.iloc[i, 4]
    
    print('Predicted calories: ', resid_plus100_df.loc[resid_plus100_df.id.eq(num), 'pred'].values, '/',
          "Group's calories  : ", group_df.loc[group_df.Age_Bucketed.eq(age) &  group_df.Duration_Bucketed.eq(duration) & 
                                  group_df.BMI_Bucketed.eq(bmi) & group_df.BMR_Bucketed.eq(bmr), 'Calories'].values)
    print('Predicted calories: ', resid_plus100_df.loc[resid_plus100_df.id.eq(num), 'pred'].values, '/',
          'Target calories   : ', train_df.loc[train_df.id.eq(num),  'Calories'].values, '\n')

    resid_plus100_df.loc[resid_plus100_df.id.eq(num), 'group_calories'] = group_df.loc[group_df.Age_Bucketed.eq(age) &  
                                                                          group_df.Duration_Bucketed.eq(duration) & 
                                                                          group_df.BMI_Bucketed.eq(bmi) & 
                                                                          group_df.BMR_Bucketed.eq(bmr), 'Calories'].values

display(resid_plus100_df)


# resid_minus100_df
print('Rasiduals are less than minus 100', '\n')

resid_minus_id = resid_minus100_df['id']
temp_df = train_df.loc[train_df.id.isin(resid_minus_id), ['id', 'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                                          'Body_Temp', 'Calories', 'Age_Bucketed', 'Duration_Bucketed', 'BMI',
                                                          'BMR', 'Healty_Weight', 'Diff_Weight', 'Over_Weight']]
display(temp_df)
print('')
temp_male_df = temp_df.loc[temp_df.Sex.eq(0)]
print('Male', ' count:', temp_male_df.shape[0])

if temp_male_df.shape[0] != 0:

    for i in range(2, 8):
        col = temp_male_df.columns[i]
        min_ = temp_male_df[col].min()
        max_ = temp_male_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
    for i in range(11, 13):
        col = temp_male_df.columns[i]
        min_ = temp_male_df[col].min()
        max_ = temp_male_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
    print('')
    print('Training data Male <describe>:')
    print(train_df.loc[train_df.Sex.eq(0), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                            'Body_Temp', 'BMI', 'BMR']].describe().T)
print('')
temp_female_df = temp_df.loc[temp_df.Sex.eq(1)]
print('Female', ' count:', temp_female_df.shape[0])

if temp_female_df.shape[0] != 0:

    for i in range(2, 8):
        col = temp_female_df.columns[i]
        min_ = temp_female_df[col].min()
        max_ = temp_female_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
    for i in range(11, 13):
        col = temp_female_df.columns[i]
        min_ = temp_female_df[col].min()
        max_ = temp_female_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
        
    print('')
    print('Training data Female <describe>:')
    print(train_df.loc[train_df.Sex.eq(1), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                        'Body_Temp', 'BMI', 'BMR']].describe().T)
    
del temp_df, temp_male_df, temp_female_df





# resid_plus100_df
print('Residuals are more than 100', '\n')

resid_plus_id = resid_plus100_df['id']
temp_df = train_df.loc[train_df.id.isin(resid_plus_id), ['id', 'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                                         'Body_Temp', 'Calories', 'Age_Bucketed', 'Duration_Bucketed', 'BMI',
                                                         'BMR', 'Healty_Weight', 'Diff_Weight', 'Over_Weight']]
display(temp_df)
print('')
temp_male_df = temp_df.loc[temp_df.Sex.eq(0)]
print('Male', ' count:', temp_male_df.shape[0])

if temp_male_df.shape[0] != 0:
    for i in range(2, 8):
        col = temp_male_df.columns[i]
        min_ = temp_male_df[col].min()
        max_ = temp_male_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
    for i in range(11, 13):
        col = temp_male_df.columns[i]
        min_ = temp_male_df[col].min()
        max_ = temp_male_df[col].max()
        print(f'{col} min: {min_} max: {max_}')

    print('')
    print('Training data Male <describe>:')
    print(train_df.loc[train_df.Sex.eq(0), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                            'Body_Temp', 'BMI', 'BMR']].describe().T)

print('')
temp_female_df = temp_df.loc[temp_df.Sex.eq(1)]
print('Female', ' count:', temp_female_df.shape[0])

if temp_female_df.shape[0] != 0:

    for i in range(2, 8):
        col = temp_female_df.columns[i]
        min_ = temp_female_df[col].min()
        max_ = temp_female_df[col].max()
        print(f'{col} min: {min_} max: {max_}')
    for i in range(11, 13):
        col = temp_female_df.columns[i]
        min_ = temp_female_df[col].min()
        max_ = temp_female_df[col].max()
        print(f'{col} min: {min_} max: {max_}')


print('Training data female <describe>:')
print(train_df.loc[train_df.Sex.eq(1), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
                                        'Body_Temp', 'BMI', 'BMR']].describe().T)

del temp_df, temp_male_df, temp_female_df


# BMI < 18.5           low weight
# 18.5 <_ BMI < 25.0   normal weight
# 25.0 <_ BMI < 30.0   Pre-obese
# 30.0 <_ BMI < 35.0   Obese class I
# 35.0 <_ BMI < 40.0   Obese class II
# 40.0 <_ BMI          Obese class III

# Male 
# 18～29 1530  30～49 1530  50～64 1480  65～74 1400  75以上 1280
# Female
# 18～29 1110  30～49 1160  50～64 1110  65～74 1080  75以上 1010


