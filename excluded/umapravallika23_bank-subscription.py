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


import pandas as pd
import numpy as np


df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


df.head()


df.columns


df.shape


df.info()


df.describe()


df.value_counts()


df.isnull().sum()


df.head()


df.dtypes


for col in df.columns : 
    if df[col].dtype==object:
        print(col,end='\n')


for col in df.columns :
    if df[col].dtype==object:
        print(f"Column {col} has unique values : {df[col].unique()}")


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


for col in df.columns :
    if df[col].dtype==object:
        df[col]=le.fit_transform(df[col])


df.head()


X=df.drop("y",axis=1)


Y=df["y"]


print(X.shape,Y.shape)


X,Y


from sklearn.linear_model import LinearRegression


model=LinearRegression()


model.fit(X,Y)


test_df=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


test_df.head()


for col in test_df.columns :
    if test_df[col].dtype==object:
        test_df[col]=le.fit_transform(test_df[col])


test_df.head()


pred=model.predict(test_df)


submission=pd.DataFrame({
    "id":test_df.id,
    "y":pred
})
submission.to_csv("Submission.csv",index=False)


output=pd.read_csv("Submission.csv")


print(output.shape)

