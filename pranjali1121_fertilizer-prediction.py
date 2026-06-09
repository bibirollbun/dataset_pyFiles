import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import seaborn as sns
import matplotlib.pyplot as plt


train_data=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
submissions=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train_data.head()


train_data['Fertilizer Name'].value_counts()


train_data.shape


train_data.info()


train_data.isnull().sum()


#checking summary statistics of the dataset
train_data.describe()


for col in train_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure()
    sns.histplot(train_data[col], kde=True)
    plt.title(f'Distribution of {col}')

