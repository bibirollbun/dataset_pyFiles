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


df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")


df.shape


df.sample(10)


df.isnull().sum()


df.describe()


import matplotlib.pyplot as plt
import seaborn as sns


# Boxplot for the Season column
sns.boxplot(df['Season'])


# Boxplot for the TeamID columns
sns.boxplot(df['TeamID'])


sns.scatterplot(df['Season'])


sns.scatterplot(df['TeamID'])


df1 = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')


df1.shape


df1.head()


sns.scatterplot(df1['Season'])


sns.scatterplot(df['TeamID'])


sns.boxplot(df1['Season'])


sns.boxplot(df1['TeamID'])


pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')


pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')


d1 = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")


d1.head(5)


d1['ID']


d1['Date'] = d1['ID'].str.split('_',expand=True)[0]


d1['Date']


d1['TeamA'] = d1['ID'].str.split('_',expand=True)[1]


d1['TeamA']


d1['TeamB'] = d1['ID'].str.split('_',expand=True)[2]


d1['TeamB']


d2 = d1[['Date','TeamA','TeamB','Pred']]


d2.to_csv("SampleSubmissionStage2Extraction.csv")

