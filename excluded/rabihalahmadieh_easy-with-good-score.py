# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
X = df.drop(['diagnosed_diabetes'],axis =1)


y = df['diagnosed_diabetes']


cat_features = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status']


model = CatBoostClassifier(eval_metric="AUC")

model.fit(X,y, cat_features = cat_features,verbose=100)


df1 =pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
X1 = df1
y1 = model.predict_proba(X1)[:,1]
answer = pd.DataFrame({'id':df1['id'],'diagnosed_diabetes':y1})
answer.to_csv('/kaggle/working/submission.csv',index = False)




