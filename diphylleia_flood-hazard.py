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


train=pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')


train.isna().sum()


train.duplicated().sum()


train.info()


train.describe()


train.columns


train['FloodProbability'].min()


# linear regression clipped should work


corr_matrix=train.corr()


import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(corr_matrix)


from sklearn.linear_model import LinearRegression



corr_matrix


model=LinearRegression()


X=train.drop(['id','FloodProbability'],axis=1)
y=train['FloodProbability']


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


model.fit(X_train,y_train)



y_pred = model.predict(X_test)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")


test.head()


test=test.drop('id',axis=1)


y_pred_df_test=model.predict(test)


test_submission=pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')


submission=test_submission[['id']].copy()
submission['FloodProbability']=y_pred_df_test


submission.head()


submission.to_csv('submission.csv', index=False)





