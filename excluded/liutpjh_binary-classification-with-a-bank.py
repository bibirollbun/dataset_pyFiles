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


from fastai.tabular.all import *
df =pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df =pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sam_df =pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


test_df.head()


df.head()


procs=[Categorify,FillMissing,Normalize]
cont,cat=cont_cat_split(df,1,dep_var='y')
path=Path('/kaggle/input/playground-series-s5e8')
dls=TabularDataLoaders.from_df(
    df,path,procs=procs,
    cat_names=cat,cont_names=cont,
    y_names='y',
    y_block=CategoryBlock(),
    bs=1024
)
learn =tabular_learner(dls,metrics=RocAucBinary(),model_dir='/kaggle/working/models')


import matplotlib.pyplot as plt
age_buy = df.groupby('age')['y'].mean()
plt.figure(figsize=(8,4))
plt.plot(age_buy.index,age_buy.values)
plt.xlabel('Age')
plt.ylabel('Proportion of y=1')
plt.title('Age vs. Probabilty of y=1')
plt.show()


# 把余额分成10档
df['balance_bin'] = pd.qcut(df['balance'], 10, duplicates='drop')

balance_buy = df.groupby('balance_bin')['y'].mean()

plt.figure(figsize=(8,4))
balance_buy.plot(kind='bar')
plt.xticks(rotation=45)
plt.ylabel('Proportion of y=1')
plt.title('Balance vs. Probability of y=1')
plt.show()


learn.lr_find()
learn.fit_one_cycle(5,1e-3)
dl_test=learn.dls.test_dl(test_df)
preds,_=learn.get_preds(dl=dl_test)
final_preds=preds[:,1]


sub =pd.read_csv(path/'sample_submission.csv')
sub['y']=final_preds
sub.to_csv('subssion.csv',index=False)




