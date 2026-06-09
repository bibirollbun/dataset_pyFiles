# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e11'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv',index_col=0)


df.head()


df.columns


df.describe()


df.shape


df.info()


categorical_cols=df.select_dtypes('object').columns


categorical_cols


df[categorical_cols].head()


for i in categorical_cols:
    print(i)
    print(df[i].nunique())


import seaborn as sns
import matplotlib.pyplot as plt


for i in categorical_cols:
    plt.figure(figsize=(12,6))
    sns.countplot(data=df,x=i,hue='loan_paid_back')
    plt.title(f'loan Payback Count by {i}')
    plt.xlabel(f'{i}')
    plt.ylabel('Number of People')
    plt.legend(title='Loan Paid Back')
    plt.show()


from sklearn.preprocessing import OrdinalEncoder


oe=OrdinalEncoder()


df[categorical_cols]=oe.fit_transform(df[categorical_cols])


df[categorical_cols].head()


numerical_cols=df.select_dtypes('float64','int64').columns


numerical_cols


df[numerical_cols].head()


for i in numerical_cols:
        print(i)
        print(df[i].nunique())


for i in numerical_cols:
    print(i)
    sns.boxplot(data=df[i])
    plt.title(f'boxplot of {i}')
    plt.show()


df.corr()


corr_matrix=df.corr()


plt.figure(figsize=(12,6))
sns.heatmap(corr_matrix)
plt.show()


x=df.drop('loan_paid_back',axis=1)
y=df['loan_paid_back']


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.head()


from xgboost import XGBClassifier


xgb_model=XGBClassifier()


xgb_model.fit(x_train,y_train)


y_pred=xgb_model.predict(x_test)


from sklearn.metrics import accuracy_score


accuracy_score(y_pred,y_test)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv',index_col=0)


df_test.head()


df.shape


df.isnull().sum()


df_test[categorical_cols]=oe.transform(df_test[categorical_cols])


df_test.head()


test_pred=xgb_model.predict_proba(df_test)


test_pred


max_probs = np.max(test_pred, axis=1)


max_probs


submission_df=pd.DataFrame({'loan_paid_back':max_probs},index=df_test.index)


submission_df.head()


submission_df.to_csv('submission.csv')


pd.read_csv('/kaggle/working/submission.csv')




