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
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


SAMPLE_SIZE=100
TARGET_COL='diagnosed_diabetes'
CAT=[]
NUM=[]
T=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv').drop(columns=['id']).sample(SAMPLE_SIZE)
for col in T.columns:
    if col==TARGET_COL:continue
    elif T[col].nunique()<10:CAT.append(col)
    else:NUM.append(col)
T


sns.pairplot(T[NUM+[TARGET_COL]],hue=TARGET_COL)


for col in CAT:
    sns.countplot(T,x=col,hue=TARGET_COL)
    plt.title(col)
    plt.show()


for col in NUM:
    sns.kdeplot(T[col])
    plt.title(col)
    plt.show()


for col in NUM:
    plt.boxplot(T[col])
    plt.title(col)
    plt.show()


sns.heatmap(T[NUM+[TARGET_COL]].corr())


for col in NUM+[TARGET_COL]:
    sns.ecdfplot(data=T, x=col)
    plt.title(col)
    plt.show()


for c1 in NUM:
    for c2 in NUM:
        if c1==c2:continue
        plt.scatter(T[c1],T[c2],c=T[TARGET_COL])
        plt.title(f'x = {c1}   y = {c2}')
        plt.show()


for col in NUM:
    if col==TARGET_COL:
        sns.kdeplot(T[TARGET_COL])
        plt.title(col)
        plt.show()
        continue
    plt.scatter(T[col],T[TARGET_COL])
    plt.title(col)
    plt.show()


t=pd.read_csv('/kaggle/input/plays12/submission (58).csv')
t.to_csv('submission.csv',index=False)
t




