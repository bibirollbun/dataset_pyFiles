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

#    torch.backends.cudnn.deterministic = True 
#    torch.use_deterministic_algorithms = True


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# Check for duplicates in the datasets

print(f'Training data duplicates: {train_df.duplicated().sum()}')
print(f'Test data duplicates    :  {test_df.duplicated().sum()}')


# Count number of distinct elements in specified axis

print('Training data:')
for col in train_df.columns:
    print(' ', col, ':', train_df[col].nunique())

print('')
print('Test data:')
for col in test_df.columns:
    print(' ', col, ':', test_df[col].nunique())


# Check for difference values between training data and test data.

print('Values of Sex:        ', set(train_df.Sex.unique()) == set(test_df.Sex.unique()))
print('Values of Age:        ', set(train_df.Age.unique()) == set(test_df.Age.unique()))
print('Values of Height:     ', set(train_df.Height.unique()) == set(test_df.Height.unique()))
print('Values of Weight:     ', set(train_df.Weight.unique()) == set(test_df.Weight.unique()))
print('Values of Heart_Rate: ', set(train_df.Heart_Rate.unique()) == set(test_df.Heart_Rate.unique()))
print('Values of Body_Temp:  ', set(train_df.Body_Temp.unique()) == set(test_df.Body_Temp.unique()))



all_df = pd.concat([train_df, test_df])
all_df.Sex.unique()


# LabelEncoding

all_df['Sex'] = all_df['Sex'].map({'male': 0, 'female': 1})


# Male

all_df.loc[all_df.Sex.eq(0)].describe().T


# Female

all_df.loc[all_df.Sex.eq(1)].describe().T


# Bucket age

boundaries_age = torch.tensor([all_df.Age.quantile(0.25), 
                               all_df.Age.quantile(0.5),
                               all_df.Age.quantile(0.75)])
v = torch.tensor(all_df['Age'].values)

all_df['Age_Bucketed'] = torch.bucketize(v, boundaries_age, right=True)


# Bucket Duration

boundaries_du = torch.tensor([all_df.Duration.quantile(0.25),
                              all_df.Duration.quantile(0.5),
                              all_df.Duration.quantile(0.75)])
v = torch.tensor(all_df['Duration'].values)

all_df['Duration_Bucketed'] = torch.bucketize(v, boundaries_du, right=True)


# BMI: Body Mass Indes  
# healty values: 18.5~24.9

all_df['BMI'] = np.round(all_df['Weight'] / ((all_df['Height']/100)**2), 2)


# Bucket BMI　by gender

all_df['BMI_Bucketed'] = 0

# Male
boundaries_bmi = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'BMI'].quantile(0.75)])
v = torch.tensor(all_df['BMI'].values)

all_df['BMI_0'] = torch.bucketize(v, boundaries_bmi, right=True)
all_df.loc[all_df.Sex.eq(0), 'BMI_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'BMI_0']


# Female
boundaries_bmi = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'BMI'].quantile(0.75)])
v = torch.tensor(all_df['BMI'].values)

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


# Bucket BMR by gender

all_df['BMR_Bucketed'] = 0

# Male
boundaries_bmr = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'BMR'].quantile(0.75)])
v = torch.tensor(all_df['BMR'].values)

all_df['BMR_0'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(0), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'BMR_0']

# Female
boundaries_bmr = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'BMR'].quantile(0.75)])
v = torch.tensor(all_df['BMR'].values)

# 1026.38~, 1296.1~, 1368.24~, 1443.42~
all_df['BMR_1'] = torch.bucketize(v, boundaries_bmr, right=True)
all_df.loc[all_df['Sex'].eq(1), 'BMR_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'BMR_1']

all_df = all_df.drop(['BMR_0', 'BMR_1'], axis=1)


# Healty_Weight = Height/100 * Height/100 * 22

all_df['Healty_Weight'] = np.round(((all_df['Height']/100) ** 2) * 22, 2)


# Bucket Healty Weight by gender

all_df['Healty_Weight_Bucketed'] = 0

# Male
boundaries_hel = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(0), 'Healty_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Healty_Weight'].values)

all_df['hel_0'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'hel_0']


# Female
boundaries_hel = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.25), 
                               all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.5), 
                               all_df.loc[all_df.Sex.eq(1), 'Healty_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Healty_Weight'].values)

all_df['hel_1'] = torch.bucketize(v, boundaries_hel, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Healty_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'hel_1']


all_df = all_df.drop(['hel_0', 'hel_1'], axis=1)


# Difference between 'Weight' and 'Healty_Weight'

all_df['Diff_Weight'] = np.round(all_df['Weight'] - all_df['Healty_Weight'], 2)


# 'Over_Weight':

all_df['Over_Weight'] = all_df['Diff_Weight']
all_df.loc[all_df['Diff_Weight'] <=0, 'Over_Weight'] = 0


all_df.loc[all_df.Sex.eq(0)]['Over_Weight'].describe().T


all_df.loc[all_df.Sex.eq(1)]['Over_Weight'].describe().T


# Bucket Over_Weight by gender

all_df['Over_Weight_Bucketed'] = 0

# Male
boundaries_ow = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Over_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Over_Weight'].values)

all_df['ow_0'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(0), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(0), 'ow_0']


# Female
boundaries_ow = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Over_Weight'].quantile(0.75)])
v = torch.tensor(all_df['Over_Weight'].values)

all_df['ow_1'] = torch.bucketize(v, boundaries_ow, right=True)
all_df.loc[all_df['Sex'].eq(1), 'Over_Weight_Bucketed'] = all_df.loc[all_df['Sex'].eq(1), 'ow_1']


all_df = all_df.drop(['ow_0', 'ow_1'], axis=1)


all_df.columns


# Group: 'Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed', (Training data)

df = all_df.loc[~all_df['Calories'].isnull()].groupby(['Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed'])['Calories'].agg('mean')

group_df = pd.DataFrame(df, columns=['Calories']).reset_index()
group_df.columns = ['Age_Bucketed', 'Duration_Bucketed', 'BMI_Bucketed', 'BMR_Bucketed', 'Calories']
group_df.head(3)


group_df.shape


group_df.sort_values(by='Calories', ascending=False).head(60)


group_df.sort_values(by='Calories', ascending=True).head(60)


# Group1: 'Age_Bucketed', 'Duration_Bucketed'  (Training data by gender)

male_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(0))].groupby(['Age_Bucketed', 'Duration_Bucketed'])['Calories'].agg('mean')
female_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(1))].groupby(['Age_Bucketed', 'Duration_Bucketed'])['Calories'].agg('mean')

group1_male_df = pd.DataFrame(male_df, columns=['Calories']).reset_index()
group1_male_df.columns = ['Age_Bucketed', 'Duration_Bucketed', 'Calories']

group1_female_df = pd.DataFrame(female_df, columns=['Calories']).reset_index()
group1_female_df.columns = ['Age_Bucketed', 'Duration_Bucketed', 'Calories']


group1_male_df.sort_values(by='Calories', ascending=False)


group1_female_df.sort_values(by='Calories', ascending=False)


all_df.loc[all_df.Sex.eq(0)]['Height'].describe().T


all_df.loc[all_df.Sex.eq(1)]['Height'].describe().T


# Height　by gender

all_df['Height_Bucketed'] = 0

# Male
boundaries_he = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Height'].quantile(0.75)])
v = torch.tensor(all_df['Height'].values)

all_df['he_0'] = torch.bucketize(v, boundaries_he, right=True)
all_df.loc[all_df.Sex.eq(0), 'Height_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'he_0']


# Female
boundaries_he = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Height'].quantile(0.75)])
v = torch.tensor(all_df['Height'].values)

all_df['he_1'] = torch.bucketize(v, boundaries_he, right=True)
all_df.loc[all_df.Sex.eq(1), 'Height_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'he_1']

all_df = all_df.drop(['he_0', 'he_1'], axis=1)


all_df.loc[all_df.Sex.eq(0)]['Weight'].describe().T


all_df.loc[all_df.Sex.eq(1)]['Weight'].describe().T


# Weight　by gender

all_df['Weight_Bucketed'] = 0

# Male
boundaries_we = torch.tensor([all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(0), 'Weight'].quantile(0.75)])
v = torch.tensor(all_df['Weight'].values)

all_df['we_0'] = torch.bucketize(v, boundaries_we, right=True)
all_df.loc[all_df.Sex.eq(0), 'Weight_Bucketed'] = all_df.loc[all_df.Sex.eq(0), 'we_0']


# Female
boundaries_we = torch.tensor([all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.25), 
                              all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.5), 
                              all_df.loc[all_df.Sex.eq(1), 'Weight'].quantile(0.75)])
v = torch.tensor(all_df['Weight'].values)

# 50~, 59~, 63~, 68~
all_df['we_1'] = torch.bucketize(v, boundaries_we, right=True)
all_df.loc[all_df.Sex.eq(1), 'Weight_Bucketed'] = all_df.loc[all_df.Sex.eq(1), 'we_1']

all_df = all_df.drop(['we_0', 'we_1'], axis=1)


# Group2: 'Height_Bucketed', 'Weight_Bucketed'  (Training data by gender)

male_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(0))].groupby(['Height_Bucketed', 'Weight_Bucketed'])['Calories'].agg('mean')
female_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(1))].groupby(['Height_Bucketed', 'Weight_Bucketed'])['Calories'].agg('mean')

group2_male_df = pd.DataFrame(male_df, columns=['Calories']).reset_index()
group2_male_df.columns = ['Height_Bucketed', 'Weight_Bucketed', 'Calories']

group2_female_df = pd.DataFrame(female_df, columns=['Calories']).reset_index()
group2_female_df.columns = ['Height_Bucketed', 'Weight_Bucketed', 'Calories']


group2_male_df.sort_values(by='Calories', ascending=False)


group2_female_df.sort_values(by='Calories', ascending=False)


all_df.columns


all_df['Height_cat'] = all_df['Height_Bucketed']
all_df['Height_cat'] = all_df['Height_cat'].astype('str')
all_df['Weight_cat'] = all_df['Weight_Bucketed']
all_df['Weight_cat'] = all_df['Weight_cat'].astype('str')

all_df['Height_Weight'] = all_df['Height_cat'] + '_' + all_df['Weight_cat']
all_df = all_df.drop(['Height_cat', 'Weight_cat'], axis=1)


all_df.head(3)


male_mapping = {
    '0_3': 3,
    '0_2': 3,
    '0_1': 3,
    '2_3': 3,
    '1_2': 2, 
    '1_3': 2, 
    '1_1': 2, 
    '3_3': 2, 
    '2_2': 1,
    '2_1': 1,
    '0_0': 1, 
    '1_0': 1, 
    '3_2': 0, 
    '3_1': 0,
    '2_0': 0,
    '3_0': 0,
}
female_mapping = {
    '0_1': 3,
    '0_2': 3,
    '0_0': 3,
    '1_1': 3,
    '2_1': 2,
    '1_0': 2, 
    '1_2': 2, 
    '2_0': 2, 
    '2_2': 1, 
    '0_3': 1,
    '2_3': 1,
    '1_3': 1, 
    '3_3': 0, 
    '3_2': 0, 
    '3_1': 0, 
    '3_0': 0,
}

all_df.loc[all_df.Sex.eq(0), 'Height_Weight'] = all_df.loc[all_df.Sex.eq(0), 'Height_Weight'].map(male_mapping)
all_df.loc[all_df.Sex.eq(1), 'Height_Weight'] = all_df.loc[all_df.Sex.eq(1), 'Height_Weight'].map(female_mapping)



all_df.head(3)


# Group3: 'Duration_Bucketed', 'Weight_Bucketed'  (Training data by gender)

male_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(0))].groupby(['Duration_Bucketed', 'Weight_Bucketed'])['Calories'].agg('mean')
female_df = all_df.loc[(~all_df['Calories'].isnull()) & (all_df.Sex.eq(1))].groupby(['Duration_Bucketed', 'Weight_Bucketed'])['Calories'].agg('mean')

group3_male_df = pd.DataFrame(male_df, columns=['Calories']).reset_index()
group3_male_df.columns = ['Duration_Bucketed', 'Weight_Bucketed', 'Calories']

group3_female_df = pd.DataFrame(female_df, columns=['Calories']).reset_index()
group3_female_df.columns = ['Duration_Bucketed', 'Weight_Bucketed', 'Calories']


group3_female_df.sort_values(by='Calories', ascending=False)


all_df['Duration_cat'] = all_df['Duration_Bucketed']
all_df['Duration_cat'] = all_df['Duration_cat'].astype('str')
all_df['Weight_cat'] = all_df['Weight_Bucketed']
all_df['Weight_cat'] = all_df['Weight_cat'].astype('str')

all_df['Duration_Weight'] = all_df['Duration_cat'] + '_' + all_df['Weight_cat']
all_df = all_df.drop(['Duration_cat', 'Weight_cat'], axis=1)


male_mapping = {
    '3_1': 3,
    '3_0': 3,
    '3_2': 3,
    '3_3': 3,
    '2_3': 2, 
    '2_1': 2, 
    '2_2': 2, 
    '2_0': 2, 
    '1_3': 1,
    '1_2': 1,
    '1_1': 1, 
    '1_0': 1, 
    '0_3': 0, 
    '0_2': 0,
    '0_1': 0,
    '0_0': 0,
}
female_mapping = {
    '3_3': 3,
    '3_2': 3,
    '3_1': 3,
    '3_0': 3,
    '2_0': 2, 
    '2_1': 2, 
    '2_2': 2, 
    '2_3': 2, 
    '1_1': 1,
    '1_2': 1,
    '1_0': 1, 
    '1_3': 1, 
    '0_1': 0, 
    '0_0': 0,
    '0_2': 0,
    '0_3': 0,
}
all_df.loc[all_df.Sex.eq(0), 'Duration_Weight'] = all_df.loc[all_df.Sex.eq(0), 'Duration_Weight'].map(male_mapping)
all_df.loc[all_df.Sex.eq(1), 'Duration_Weight'] = all_df.loc[all_df.Sex.eq(1), 'Duration_Weight'].map(female_mapping)



all_df.tail()


num_cols = ['Age', 'Height', 'Weight', 'Heart_Rate', 'Body_Temp'] 

df = all_df.iloc[:750000]
scatterplotmatrix(df[num_cols].values, figsize=(12, 10), names=num_cols, alpha=0.5)
plt.tight_layout()
plt.show()



num_cols = ['BMI', 'BMR', 'Healty_Weight', 'Diff_Weight', 'Over_Weight']

scatterplotmatrix(df[num_cols].values, figsize=(12, 10), names=num_cols, alpha=0.5)
plt.tight_layout()
plt.show()


num_cols = ['BMI', 'BMR', 'Healty_Weight', 'Diff_Weight', 'Over_Weight', 'Calories']

df = all_df.iloc[:750000]
cm = np.corrcoef(df[num_cols].values.T)
hm = heatmap(cm, row_names=num_cols, column_names=num_cols, cmap='Greens')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson transformer
pt = PowerTransformer(method='yeo-johnson')

df = all_df.iloc[:750000]

trans_cols = ['Age', 'Height', 'Body_Temp', 'Over_Weight']

pt.fit(df[trans_cols])
df[trans_cols] = pt.transform(df[trans_cols])

scatterplotmatrix(df[trans_cols].values, figsize=(12, 10), names=trans_cols, alpha=0.5)
plt.tight_layout()
plt.show()


all_df[['Age', 'Height', 'Body_Temp', 'Over_Weight']].describe().T


from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson transformer
# pt = PowerTransformer(method='yeo-johnson')

# Box-Cox transformer
pt = PowerTransformer(method='box-cox')

df = all_df.iloc[:750000]

trans_cols = ['Age', 'Height', 'Body_Temp']

pt.fit(df[trans_cols])
df[trans_cols] = pt.transform(df[trans_cols])

scatterplotmatrix(df[trans_cols].values, figsize=(12, 10), names=trans_cols, alpha=0.5)
plt.tight_layout()
plt.show()


num_cols = ['Age', 'Height', 'Weight', 'Heart_Rate', 'Body_Temp', 'BMI', 'BMR', 'Healty_Weight', 'Diff_Weight', 'Over_Weight', 'Calories']
df = all_df.iloc[:750000]

sns.set(font_scale=0.6)
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), linewidth=0.2,
            annot=True, annot_kws={"fontsize": 10}, cmap='Greens'
)
plt.show()

