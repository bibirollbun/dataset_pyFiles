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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection 




sample_sub=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample_sub.columns


df=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


df.head()


df['kfold']=-1
df=df.sample(frac=1,random_state=42).reset_index(drop=True)
kf=model_selection.KFold(n_splits=5,shuffle=True,random_state=42)
for f,(t_,v_) in enumerate(kf.split(X=df)):
    df.loc[v_,'kfold']=f



df.kfold.value_counts()


df.to_csv('trains_fold.csv',index=False)

