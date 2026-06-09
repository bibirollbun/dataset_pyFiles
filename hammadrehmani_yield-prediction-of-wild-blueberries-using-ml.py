# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


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
import matplotlib.pyplot as plt
import seaborn as sns


ss_df = pd.read_csv('/kaggle/input/pascmlsig/sample_submission.csv')
ss_df.head()


tt_df = pd.read_csv('/kaggle/input/pascmlsig/test.csv')
tt_df.head()


tn_df = pd.read_csv('/kaggle/input/pascmlsig/train.csv')
tn_df.head()


print(f"The number of Rows in a dataset is {tn_df.shape[0]}")
print(f"The number of columns in a dataset is {tn_df.shape[1]}")


X = tn_df.drop(columns = 'yield',axis=1)
X.head()


Y = tn_df['yield']
Y.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


tt_data_id = tt_df['id']


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
tt_df = ss.fit_transform(tt_df)


lr = LinearRegression()
lr.fit(X_train,Y_train)


y_pred = lr.predict(X_test)

print('R2 score',r2_score(Y_test,y_pred))
print('MAE score',mean_absolute_error(Y_test,y_pred))
print('mean_squared_error',mean_squared_error(Y_test,y_pred))


output = lr.predict(tt_df)
output


submission_df = pd.DataFrame({
    'id': tt_data_id,
    'yield': output
})


submission_df


submission_df.to_csv('submission.csv', index=False)

print('The file is ready for submission!')




