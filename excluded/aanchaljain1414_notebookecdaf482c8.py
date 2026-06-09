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


import pandas as pd



df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')



df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')



df_train


df_train.columns


df_test.columns


target='diagnosed_diabetes'


y = df_train[target]
train_features = df_train.drop(columns=[target])


df=pd.concat([train_features,df_test],axis=0,ignore_index=True)


df


df.isnull().sum()


df.info()


df['gender']


df['gender'].value_counts()


df['Gender_encoded'] = df['gender'].map({'Male': 2, 'Female': 1,'Other': 0})


df.drop(columns=['gender'],inplace=True)


df['Gender_encoded'].value_counts


df['Gender_encoded']=df['Gender_encoded'].astype(int)


df.select_dtypes(include='object').columns



df['ethnicity']


df['ethnicity'].value_counts()


df['ethnicity'] = df['ethnicity'].str.lower().str.strip()

df = pd.get_dummies(
    df,
    columns=['ethnicity'],
    prefix='ethnicity',
    drop_first=True
)



df['education_level'].value_counts()


edu_map = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

df['education_level_encoded'] = df['education_level'].map(edu_map)
df.drop(columns=['education_level'], inplace=True)



df['income_level'].value_counts()


income_map = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}

df['income_level_encoded'] = df['income_level'].map(income_map)
df.drop(columns=['income_level'], inplace=True)



df['smoking_status'].value_counts()


df['employment_status'].value_counts()


smoking_map = {
    'Never': 0,
    'Former': 1,
    'Current': 2
}

df['smoking_status_encoded'] = df['smoking_status'].map(smoking_map)
df.drop(columns=['smoking_status'], inplace=True)



df['employment_status'] = df['employment_status'].str.lower().str.strip()

df = pd.get_dummies(
    df,
    columns=['employment_status'],
    prefix='employment_status',
    drop_first=True
)




