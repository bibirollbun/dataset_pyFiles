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


train= pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.columns.to_list()


test= pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.columns.to_list()


x_train= train.drop(columns=['id','Calories'])
y_train= train['Calories']


x_test= test.drop(columns='id')


x_train.describe()


x_train.info()


cat_col= x_train.select_dtypes(include='object').columns


x_train['Sex'].value_counts()


x_train.isnull().any()


import seaborn as sns
import matplotlib.pyplot as plt

corr= x_train.corr(numeric_only= True)
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot= True, cmap= 'coolwarm')


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

preprocessing= ColumnTransformer(
    transformers=[
        ('standardscaler', StandardScaler(), ['Height', 'Weight', 'Heart_Rate']),
        ('minmaxscaler', MinMaxScaler(), ['Duration']),
        ('ohe', OneHotEncoder(drop='if_binary'), ['Sex'])
    ],
    remainder= 'passthrough'
)

pipeline= Pipeline([
    ('preprocessor', preprocessing),
    ('model', RandomForestRegressor(n_estimators=10, random_state=0))
])

pipeline.fit(x_train, y_train)



prediction= pipeline.predict(x_test)


submission= pd.DataFrame({
    'id': test['id'],
    'Calories': prediction
})


print(submission.head())


submission.to_csv('Submission.csv', index= False)




