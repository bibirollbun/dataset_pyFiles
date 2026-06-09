# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Basic Libraries
import pandas as pd
import numpy as np

# Data Viz 
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings ignore
import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn')
sns.set_style('darkgrid')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train.head()


train.shape


summary_train = pd.DataFrame({
    'Data_type' : train.dtypes.values,
    'Null_Percentage' : round((train.isna().sum()/train.shape[0])*100, 2),
    'Total_Values_Avail' : train.notnull().sum().values,
    'Unique_Values' : train.nunique()
})
summary_train


plt.figure(figsize=(10, 4))
train.Personality.value_counts(dropna=False).plot.pie(autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        legend=False, 
        explode = [0.05, 0.05])
plt.title("Personality")
plt.ylabel("")
plt.tight_layout()


cols = ['Stage_fear', 'Drained_after_socializing']

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

for i, col in enumerate(cols):
    
    # Pie chart in [row, 0]
    data = train[col].value_counts(dropna=False)
    explode_val = [0.05] * len(data)

    data.plot.pie(
        ax=axes[i, 0],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        legend=False,
        explode=explode_val
    )
    axes[i, 0].set_ylabel('')
    axes[i, 0].set_title(f'{col} Distribution')

    # Countplot in [row, 1]
    sns.countplot(ax=axes[i, 1], data=train, x=col, hue='Personality')
    axes[i, 1].set_title(f'{col} vs Personality')
    axes[i, 1].set_xlabel('')

plt.tight_layout()


num_col = list(train.select_dtypes(exclude='object'))
train[num_col].hist(figsize=(15, 10))
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
axes = axes.flatten()
for i, col in enumerate(num_col):
    sns.boxplot(data=train, x='Personality', y=col, ax=axes[i])
    axes[i].set_title(f'{col} vs Personality')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    
plt.tight_layout()







