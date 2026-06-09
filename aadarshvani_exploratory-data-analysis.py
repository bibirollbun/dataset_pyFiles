# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Training and Testing data 
train_df  =pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Preliminary data overview
train_df.head(5)


train_df.tail(5)


# Shape of the training data
print('Shape of the data', train_df.shape)


# Basic Information about the data 
train_df.info()


# Basic Discription of numberical features in the data 

train_df[['Compartments', 'Weight Capacity (kg)', 'Price']].describe()


# Lets clean the data for missing values

train_df.isnull().sum()


# Droping the null values 
train_df.dropna(inplace = True)


# Lets check if all null values have been removed and also the shape of the data
print('New shape of the data ', train_df.shape)

train_df.isnull().sum()


ids = train_df['id']
X_train = train_df.drop(columns= ['id', 'Price'])
y_train = train_df['Price']


# Extracting name of non numerical features to encode



# Lets encode the data set, now here we have to label both Training and Testing data for consistency in the output 

# Lets import necessary libarareis and encoders 

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

train_data = encoder.fit_transform(train_df)
test_data = encoder.transform(test_df)










