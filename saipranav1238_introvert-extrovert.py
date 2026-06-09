# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

input_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

print(input_df.head())
print('printing info ',input_df.info())
print('printing description \n', input_df.describe())
print('double brackets',type(input_df[['id','Time_spent_Alone']]))
print(input_df.isna())
print('sum of na',input_df.isna().sum())
input_df=input_df.dropna()
print('sum of na',input_df.isna().sum())
encoded_df=pd.get_dummies(input_df,columns=['Stage_fear','Drained_after_socializing'], dtype=int)
print(encoded_df.head())

x_train=encoded_df.drop(['Personality'],axis=1)
y_train=encoded_df['Personality']
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_train)
print('accuracy: ',accuracy_score(y_train,y_pred))

test_df=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
encoded_test_df=pd.get_dummies(test_df,columns=['Stage_fear','Drained_after_socializing'], dtype=int)
encoded_test_df=encoded_test_df.fillna(0)
y_pred=model.predict(encoded_test_df)

submission=pd.DataFrame({'id':encoded_test_df['id'],'Personality':y_pred})
print('submission:',submission.head())

submission.to_csv('submission.csv', index=False)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

