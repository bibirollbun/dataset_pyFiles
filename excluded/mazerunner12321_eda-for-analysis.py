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


# !pip install -q kaggle
import os
import sys

def is_kaggle():
    return 'KAGGLE_URL_BASE' in os.environ

if not is_kaggle():
    !mkdir -p ~/.kaggle
    !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

# Download dataset
competition_name = 'playground-series-s5e5'
if not is_kaggle():
    !kaggle competitions download -c {competition_name}
    !unzip -q {competition_name}.zip -d {competition_name}
    data_path = f'./{competition_name}/'
else:
    # In Kaggle, data is available at this path:
    data_path = '../input/'


print("Available files:")
!ls -la {data_path}


import pandas as pd

# # train_df = pd.read_csv(f"{data_path}/train.csv")
# # test_df = pd.read_csv(f"{data_path}/test.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col='id')


train_df


def bmi_feature(df):
  df['bmi'] = df['Weight'] / (df['Height']/100)**2
  return df

train_df = bmi_feature(train_df)
test_df = bmi_feature(test_df)


train_df.info()


sns.countplot(x='Sex',data=train_df, palette='hls')


num_cols = train_df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
  plt.figure(figsize=(8,5))
  plt.subplot(1,2,1)
  sns.histplot(train_df[col],kde=True, bins=30)
  plt.title(f'Distribution of {col}')
  plt.xlabel(col)
  plt.ylabel('Frequency')

  plt.subplot(1,2,2)
  sns.boxplot(train_df[col])
  plt.title(f'Boxplot of {col}')
  plt.xlabel(col)
  plt.ylabel('Value')

  plt.tight_layout()
  plt.show()


num_cols = train_df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    plt.figure(figsize=(10, 5))

    # Histogram with Sex hue
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x=col, hue='Sex', kde=True, bins=30, multiple='layer')
    plt.title(f'Distribution of {col} by Sex')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Boxplot with Sex split
    plt.subplot(1, 2, 2)
    sns.boxplot(data=train_df, x='Sex', y=col)
    plt.title(f'Boxplot of {col} by Sex')
    plt.xlabel('Sex')
    plt.ylabel(col)

    plt.tight_layout()
    plt.show()



train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})


# Example: correlation matrix
corr_matrix = train_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={'shrink': 0.8}
)

plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





