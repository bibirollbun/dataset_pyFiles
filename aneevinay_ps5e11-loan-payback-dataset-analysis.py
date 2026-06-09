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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_path ='/kaggle/input/playground-series-s5e11/train.csv'
train=pd.read_csv(train_path)

test_path='/kaggle/input/playground-series-s5e11/test.csv'
test=pd.read_csv(test_path)


def dataset_summary (datasets):
    summary=[]
    
    for name,df,path in datasets:
        size_on_disk=os.path.getsize(path)/(1024*1024) #MB
        size_in_memory=df.memory_usage(deep=True).sum()/(1024*1024) #MB
        rows,cols=df.shape

        summary.append({
            'Dataset':name,
            'Size on disk(MB)':round(size_on_disk,2),
            'Size in memory(MB)':round(size_in_memory,2),
            '# of rows':rows,
            '# of cols':cols
        })

    return pd.DataFrame(summary)


datasets=[
    ('train',train,train_path),
    ('test',test,test_path)
]
dataset_summary(datasets)


train.head()


test.head()


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


test.duplicated().sum()


num_cols = train.select_dtypes(include='number').columns

for col in num_cols:
    sns.histplot(train[col],bins=30, kde=True)
    plt.tight_layout()
    plt.show()


for col in num_cols:
    sns.kdeplot(train[col], fill=True, bw_adjust=1)
    plt.title(f'Distribution of {col}')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()


skew_values = train[num_cols].skew().sort_values(ascending=False)
print(skew_values)


cat_cols = train.select_dtypes(include='object').columns

for col in cat_cols:
    sns.countplot(x=col,data=train)
    plt.tight_layout()
    plt.show()




